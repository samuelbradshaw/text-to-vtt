# Python standard libraries
import os
import sys
import csv
from datetime import date, datetime
import textwrap
import re

sys.stdout.write('\nStarting…\n')

# Third-party libraries
# TODO: See if whisper-jax will work and if it's faster (as claimed), when word-level timestamps are supported (https://github.com/sanchit-gandhi/whisper-jax/issues/37)
import demucs.separate
import whisperx
import requests
import librosa

# Set working directory
working_directory = os.path.abspath(os.path.dirname(__file__))

# Set output directory
output_folder = os.path.join(working_directory, '_output')
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

# Set workfiles directory
workfiles_folder = os.path.join(working_directory, '_workfiles')
workfiles_media_folder = os.path.join(workfiles_folder, 'media')
workfiles_text_folder = os.path.join(workfiles_folder, 'text')
if not os.path.exists(workfiles_folder):
  os.makedirs(workfiles_folder)
if not os.path.exists(workfiles_media_folder):
  os.makedirs(workfiles_media_folder)
if not os.path.exists(workfiles_text_folder):
  os.makedirs(workfiles_text_folder)

sys.stdout.write('Checking for input data\n')

try:
  csv_input_path = os.path.normpath(sys.argv[1])
except IndexError:
  sys.stderr.write('  Error: Input CSV or TSV file not specified.\n\n')
  sys.exit(1)

# Set default options
class defaults:
  def __init__(self):
    self.use_voice_isolation = False
    self.vtt_output_types = ['phrase']
    self.break_phrases_at = ['single_line_breaks']
    self.max_phrase_character_count = 40
    self.word_timestamps = False
    self.time_offset = -0.3

options = defaults()

# Demucs and whisperX configuration
demucs_model = 'mdx_extra'
whisperx_device = 'cpu' # TODO: For macOS, switch to mps when it's working. See https://pytorch.org/docs/stable/notes/mps.html and https://github.com/pytorch/pytorch/issues/103343
whisperx_batch_size = 4
whisperx_compute_type = 'int8'
whisperx_supported_languages = ['de', 'en', 'es', 'fr', 'it', 'ja', 'nl', 'pt', 'uk', 'zh']

# Variables for storing data
whisperx_models_by_lang = {}
input_rows = []

with open(csv_input_path, newline='') as f:
  dialect = 'excel'
  if '.tsv' in csv_input_path:
    dialect = 'excel-tab'
  csv_reader = csv.DictReader(f, dialect=dialect)
  for row in csv_reader:
    input_rows.append(row)

# Load speech models for each language
sys.stdout.write('Loading speech models\n')
for input_row in input_rows:
  if input_row.get('lang'):
    whisperx_lang = input_row.get('lang')
    if whisperx_lang not in whisperx_supported_languages:
      sys.stderr.write('  Warning: Language code \'{0}\' not supported – falling back to \'en\'.\n'.format(whisperx_lang))
      whisperx_lang = 'en'
    if whisperx_lang not in whisperx_models_by_lang:
      sys.stdout.write(f'\nWHISPERX: whisperx.load_model: {whisperx_lang}\n')
      whisperx_models_by_lang[whisperx_lang] = whisperx.load_model('large-v2', language=whisperx_lang, device=whisperx_device, compute_type=whisperx_compute_type)
      sys.stdout.write('END\n')

# Set options if they're specified in the input
def set_options(input_row):
  global options
  options = defaults()
  if input_row.get('use_voice_isolation', '') != '':
    if any(v in input_row.get('use_voice_isolation').lower() for v in ['1', 'y', 't']):
      options.use_voice_isolation = True
    else:
      options.use_voice_isolation = False
  if input_row.get('vtt_output_types', '') != '':
    vtt_output_types = []
    if 'b' in input_row.get('vtt_output_types').lower():
      vtt_output_types.append('block')
    if 'p' in input_row.get('vtt_output_types').lower():
      vtt_output_types.append('phrase')
    if 'w' in input_row.get('vtt_output_types').lower():
      vtt_output_types.append('word')
    options.vtt_output_types = vtt_output_types
  if input_row.get('break_phrases_at', '') != '':
    break_phrases_at = []
    if 'line' in input_row.get('break_phrases_at').lower():
      break_phrases_at.append('single_line_breaks')
    if 'sent' in input_row.get('break_phrases_at').lower():
      break_phrases_at.append('sentence_punctuation')
    if 'char' in input_row.get('break_phrases_at').lower():
      break_phrases_at.append('character_count')
    options.break_phrases_at = break_phrases_at
  if input_row.get('max_phrase_character_count', '') != '':
    max_phrase_character_count = int(input_row.get('max_phrase_character_count'))
    if max_phrase_character_count:
      options.max_phrase_character_count = max_phrase_character_count
  if input_row.get('word_timestamps', '') != '':
    if any(v in input_row.get('word_timestamps').lower() for v in ['1', 'y', 't']):
      options.word_timestamps = True
    else:
      options.word_timestamps = False
  if input_row.get('time_offset', '') != '':
    time_offset = float(input_row.get('time_offset'))
    if time_offset:
      options.time_offset = time_offset

# Convert seconds to a VTT timestamp
def seconds_to_vtt_timestamp(seconds):
  adjustedSeconds = max(0, seconds + options.time_offset)
  dt = datetime.utcfromtimestamp(adjustedSeconds)
  return dt.isoformat(sep='T', timespec='milliseconds').split('T')[1]

# Process input rows
for i, input_row in enumerate(input_rows):
  set_options(input_row)
  
  sys.stdout.write('\n{0} of {1}: {2}\n'.format(i + 1, len(input_rows), input_row.get('id')))
  sys.stdout.write(f'Options: {vars(options)}\n')
  
  if not input_row.get('id') or not input_row.get('title') or not input_row.get('lang'):
    sys.stderr.write(f'  Error: id, title, and lang must be provided for each input row. Skipping row.\n')

  # Get text from file if text file exists
  text_path = os.path.join(workfiles_text_folder, input_row.get('id') + '.txt')
  if os.path.isfile(text_path):
    with open(text_path, 'r') as f:
      input_row['input_text'] = f.read()
  elif input_row.get('input_text'):
    with open(text_path, 'w') as f:
      f.write(input_row.get('input_text'))
  
  # Download audio or video if needed
  media_path = None
  for root, dirs, files in os.walk(workfiles_media_folder):
    for file in files:
      filename_without_extension, extension = os.path.splitext(file)
      if filename_without_extension == input_row.get('id'):
        media_path = os.path.join(root, file)
  if not media_path and '://' in input_row.get('input_media', ''):
    media_url = input_row.get('input_media').split('?')[0].split('#')[0]
    extension = media_url.split('.')[-1]
    media_path = os.path.join(workfiles_media_folder, input_row.get('id') + '.' + extension)
    r = requests.get(media_url)
    if r.status_code == 200:
      with open(media_path, 'wb') as f:
        f.write(r.content)
    else:
      sys.stderr.write(f'  Error: Download failed (status code {r.status_code}). Skipping row.\n')
      continue
  if not media_path:
    sys.stderr.write(f'  Error: Media not provided. Skipping row.\n')
  
  # Isolate voice from audio (demucs)
  if options.use_voice_isolation:
    demucs_output_path = os.path.join(workfiles_folder, demucs_model, input_row.get('id'), 'vocals.wav')
    if not os.path.isfile(demucs_output_path):
      sys.stdout.write('  Isolating voice from audio\n')
      sys.stdout.write('\nDEMUCS: demucs.separate\n')
      demucs.separate.main(['--two-stems', 'vocals', '-n', demucs_model, '-o', workfiles_folder, media_path])
      sys.stdout.write('END\n\n')
    media_path = demucs_output_path
  
  whisperx_lang = input_row.get('lang') if input_row.get('lang') in whisperx_supported_languages else 'en'
  audio_duration = librosa.get_duration(path=media_path)
  audio = whisperx.load_audio(media_path)
  generated_transcript_data = None
  rough_segments = []
  
  if not input_row.get('input_text'):
    # Transcribe audio to text (whisperX)
    sys.stdout.write('\n  Transcribing audio to text\n')
    sys.stdout.write('\nWHISPERX: model.transcribe\n')
    generated_transcript_data = whisperx_models_by_lang[whisperx_lang].transcribe(audio, batch_size=whisperx_batch_size)
    sys.stdout.write('END\n\n')
    rough_segments = generated_transcript_data['segments']
    sys.stdout.write('\n')
  
  else:
    # Load provided transcript
    transcript = input_row.get('input_text')
    
    # Remove content bracketed with <> or [] (for example, HTML tags, [Chorus] markers, or bracketed words that were added by an editor)
    transcript = re.sub('<[^>]+?>', ' ', transcript)
    transcript = re.sub('\[[^\]]+?\]', ' ', transcript)
    
    # Break transcript up into blocks and phrases
    transcript_blocks = [b.strip() for b in transcript.split('\n\n')]
    transcript_block_phrases = []
    transcript = transcript.replace('\r', '')
    for block in transcript_blocks:
      transcript_block_phrases.append([block])
    if 'single_line_breaks' in options.break_phrases_at:
      for phrases_list in transcript_block_phrases:
        new_phrases_list = []
        for phrase in phrases_list:
          new_phrases_list.extend([p.strip() for p in phrase.split('\n')])
        phrases_list.clear()
        phrases_list += new_phrases_list
    if 'sentence_punctuation' in options.break_phrases_at:
      for phrases_list in transcript_block_phrases:
        new_phrases_list = []
        for phrase in phrases_list:
          phrases_and_punctuations = re.split(r'(\.”|\."|\.’\s|\.\'\s|\.|;|:|—|–|…|!|\?|$)', phrase)
          new_phrases_list.extend([''.join(p).strip() for p in zip(phrases_and_punctuations[0::2], phrases_and_punctuations[1::2]) if ''.join(p).strip()])
        phrases_list.clear()
        phrases_list += new_phrases_list
    if 'character_count' in options.break_phrases_at:
      for phrases_list in transcript_block_phrases:
        new_phrases_list = []
        for phrase in phrases_list:
          if len(phrase) > options.max_phrase_character_count:
            new_phrases_list.extend(textwrap.wrap(phrase, width=options.max_phrase_character_count))
          else:
            new_phrases_list.append(phrase)
        phrases_list.clear()
        phrases_list += new_phrases_list
    
    # Mark the beginning of each block with **** and the beginning of each phrase with **
    modified_transcript = ''
    short_previous_phrase = ''
    for block in transcript_block_phrases:
      modified_transcript += ' **** '
      for phrase in block:
        phrase = short_previous_phrase + ' ' + phrase
        if len(phrase) <= 5:
          short_previous_phrase = phrase
          continue
        else:
          short_previous_phrase = ''
        modified_transcript += ' ** '
        modified_transcript += phrase
    modified_transcript = modified_transcript.replace(' ****  **** ', ' **** ').replace(' **  ** ', ' ** ').strip()
    rough_segments = [{
      'text': modified_transcript,
      'start': generated_transcript_data['segments'][0]['start'] if generated_transcript_data else 0,
      'end': generated_transcript_data['segments'][-1]['end'] if generated_transcript_data else audio_duration,
    }]
    
  # Align transcript to audio
  sys.stdout.write('  Aligning text to audio\n')
  model_a, metadata = whisperx.load_align_model(language_code=whisperx_lang, device=whisperx_device)
  result = whisperx.align(rough_segments, model_a, metadata, audio, whisperx_device, return_char_alignments=False)
  segments_with_word_data = result['segments']

  # Structure timestamp data more cleanly
  sys.stdout.write('  Structuring data\n')
  blocks = []
  text_without_timestamps = ''
  for segment in segments_with_word_data:
    for word_data in segment['words']:
      if word_data['word'] == '****':
        # Prevent text without timestamps at the end of a block from being carried over to the next block
        if text_without_timestamps:
          blocks[-1]['phrases'][-1]['words'][-1]['text'] += ' ' + text_without_timestamps.strip()
          text_without_timestamps = ''
        # Add a new block
        blocks.append({
          'start_seconds': None,
          'end_seconds': None,
          'phrases': [],
        })
        continue
      if word_data['word'] == '**':
        # Prevent text without timestamps at the end of a phrase from being carried over to the next phrase
        if text_without_timestamps:
          blocks[-1]['phrases'][-1]['words'][-1]['text'] += ' ' + text_without_timestamps.strip()
          text_without_timestamps = ''
        # Add a new phrase in the current block
        blocks[-1]['phrases'].append({
          'start_seconds': None,
          'end_seconds': None,
          'words': [],
        })
        continue
      
      if 'start' in word_data and 'end' in word_data:
        # Update time info in current block and phrase
        if not blocks[-1]['start_seconds']:
          blocks[-1]['start_seconds'] = word_data['start']
        if not blocks[-1]['phrases'][-1]['start_seconds']:
          blocks[-1]['phrases'][-1]['start_seconds'] = word_data['start']
        blocks[-1]['end_seconds'] = word_data['end']
        blocks[-1]['phrases'][-1]['end_seconds'] = word_data['end']
        blocks[-1]['end_seconds'] = word_data['end']
        blocks[-1]['phrases'][-1]['end_seconds'] = word_data['end']
        
        # Add a new word in the current phrase
        blocks[-1]['phrases'][-1]['words'].append({
          'start_seconds': word_data['start'],
          'end_seconds': word_data['end'],
          'text': text_without_timestamps + word_data['word']
        })
        
        text_without_timestamps = ''
      elif word_data['word']:
        text_without_timestamps += word_data['word'] + ' '
      
  # Create VTT files
  sys.stdout.write('  Creating VTT file(s)\n')
  vtt_block = vtt_phrase = vtt_word = textwrap.dedent(f'''\
    WEBVTT - {input_row.get('title')}

    NOTE
    {{
      "_": "Generated {date.today()} by Text to VTT (https://github.com/samuelbradshaw/text-to-vtt)"
      "id": "{input_row.get('id')}",
      "lang": "{input_row.get('lang')}",
    }}
  ''')
  
  for vtt_type in options.vtt_output_types:
    # Block-based VTT
    if vtt_type == 'block':
      vtt = vtt_block
      for b, block in enumerate(blocks):
        formatted_start = seconds_to_vtt_timestamp(block['start_seconds'])
        formatted_end = seconds_to_vtt_timestamp(block['end_seconds'])
        vtt += textwrap.dedent(f'''\
          
          {{"block": {b + 1}}}
          {formatted_start} --> {formatted_end}
        ''')
        for p, phrase in enumerate(block['phrases']):
          for w, word in enumerate(phrase['words']):
            if options.word_timestamps:
              formatted_word_start = seconds_to_vtt_timestamp(word['start_seconds'])
              vtt += f'<{formatted_word_start}>'
            vtt += word['text'] + ' '
          vtt = vtt.strip() + '\n'
        
    # Phrase-based VTT
    if vtt_type == 'phrase':
      vtt = vtt_phrase
      for b, block in enumerate(blocks):
        for p, phrase in enumerate(block['phrases']):
          formatted_start = seconds_to_vtt_timestamp(phrase['start_seconds'])
          formatted_end = seconds_to_vtt_timestamp(phrase['end_seconds'])
          vtt += textwrap.dedent(f'''\
      
            {{"block": {b + 1}, "phrase": {p + 1}}}
            {formatted_start} --> {formatted_end}
          ''')
          for w, word in enumerate(phrase['words']):
            if options.word_timestamps:
              formatted_word_start = seconds_to_vtt_timestamp(word['start_seconds'])
              vtt += f'<{formatted_word_start}>'
            vtt += word['text'] + ' '
          vtt = vtt.strip() + '\n'
    
    # Word-based VTT
    if vtt_type == 'word':
      vtt = vtt_word
      vtt_type = 'word'
      for b, block in enumerate(blocks):
        for p, phrase in enumerate(block['phrases']):
          for w, word in enumerate(phrase['words']):
            formatted_start = seconds_to_vtt_timestamp(word['start_seconds'])
            formatted_end = seconds_to_vtt_timestamp(word['end_seconds'])
            vtt += textwrap.dedent(f'''\
      
            {{"block": {b + 1}, "phrase": {p + 1}, "word": {w + 1}}}
              {formatted_start} --> {formatted_end}
            ''')
            if options.word_timestamps:
              formatted_word_start = seconds_to_vtt_timestamp(word['start_seconds'])
              vtt += f'<{formatted_word_start}>'
            vtt += word['text'] + '\n'
    
    vtt_output_folder = os.path.join(output_folder, vtt_type)
    if not os.path.exists(vtt_output_folder):
      os.makedirs(vtt_output_folder)
    vtt_output_path = os.path.join(vtt_output_folder, '{0}.vtt'.format(input_row.get('id')))
    with open(vtt_output_path, 'w', encoding='utf-8') as f:
      f.write(vtt)
    
sys.stdout.write('\nDone!\n\n')
