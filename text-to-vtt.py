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
import stable_whisper
import requests
import librosa
from pydub import AudioSegment

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

# Load data and options from a CSV or TSV file
input_rows = []
sys.stdout.write('Checking for input data\n')
try:
  csv_input_path = os.path.normpath(sys.argv[1])
except IndexError:
  sys.stderr.write('  Error: Input CSV or TSV file not specified.\n\n')
  sys.exit(1)
with open(csv_input_path, newline='') as f:
  dialect = 'excel'
  if '.tsv' in csv_input_path:
    dialect = 'excel-tab'
  csv_reader = csv.DictReader(f, dialect=dialect)
  for row in csv_reader:
    input_rows.append(row)

whisperx_models = {}
stable_ts_models = {}
supported_languages = {
  'whisperx': ['ar', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru', 'te', 'tr', 'uk', 'ur', 'vi', 'zh'],
  'stable_ts': ['af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'],
}
device = 'cpu' # TODO: For macOS, switch to mps when it's working. See https://pytorch.org/docs/stable/notes/mps.html and https://github.com/pytorch/pytorch/issues/103343

def whisperx_model(size):
  if size not in whisperx_models:
    whisperx_models[size] = whisperx.load_model(size, device=device, compute_type='int8')
  return whisperx_models[size]

def stable_ts_model(size):
  if size not in stable_ts_models:
    stable_ts_models[size] = stable_whisper.load_model(size, dq=False)
  return stable_ts_models[size]
    
# Default options
class defaults:
  def __init__(self):
    self.alignment_method = 'whisperx'
    self.use_voice_isolation = False
    self.vtt_output_types = ['phrase']
    self.break_phrases_at = ['single_line_breaks']
    self.max_phrase_character_count = 40
    self.word_timestamps = False
    self.time_offset = -0.3

# Get options (based on input and defaults)
def get_options(input_row):
  options = defaults()
  if input_row.get('alignment_method', '') != '':
    if 'x' in input_row.get('alignment_method').lower():
      options.alignment_method = 'whisperx'
    else:
      options.alignment_method = 'stable_ts'
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
  return options

# Convert seconds to a VTT timestamp
def seconds_to_vtt_timestamp(seconds, time_offset):
  adjustedSeconds = max(0, seconds + time_offset)
  dt = datetime.utcfromtimestamp(adjustedSeconds)
  return dt.isoformat(sep='T', timespec='milliseconds').split('T')[1]

# Process input rows
for i, input_row in enumerate(input_rows):
  options = get_options(input_row)
  
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
      if file == input_row.get('id') + '.wav':
        media_path = os.path.join(root, file)
        break
  if not media_path and '://' in input_row.get('input_media', ''):
    media_url = input_row.get('input_media').split('?')[0].split('#')[0]
    extension = media_url.split('.')[-1]
    media_path = os.path.join(workfiles_media_folder, input_row.get('id') + '.' + extension)
    r = requests.get(media_url)
    if r.status_code == 200:
      with open(media_path, 'wb') as f:
        f.write(r.content)
      if extension != 'wav':
        # Convert to WAV format
        new_media_path = os.path.join(workfiles_media_folder, input_row.get('id') + '.wav')
        sound = AudioSegment.from_file(media_path, extension)
        sound.export(new_media_path, format='wav')
        os.remove(media_path)
        media_path = new_media_path
    else:
      sys.stderr.write(f'  Error: Download failed (status code {r.status_code}). Skipping row.\n')
      continue
  if not media_path:
    sys.stderr.write(f'  Error: Media not provided. Skipping row.\n')
  
  # Isolate voice from audio (demucs)
  if options.use_voice_isolation:
    demucs_model_type = 'mdx_extra'
    demucs_output_path = os.path.join(workfiles_folder, demucs_model_type, input_row.get('id'), 'vocals.wav')
    if not os.path.isfile(demucs_output_path):
      sys.stdout.write('  Isolating voice from audio\n')
      sys.stdout.write('\nDEMUCS: demucs.separate\n')
      demucs.separate.main(['--two-stems', 'vocals', '-n', demucs_model_type, '-o', workfiles_folder, media_path])
      sys.stdout.write('END\n\n')
    media_path = demucs_output_path
    
  lang = input_row.get('lang') if input_row.get('lang') in supported_languages[options.alignment_method] else 'en'
  
  # Load audio into whisperX if needed
  whisperx_loaded_audio = None
  if options.alignment_method == 'whisperx' or not input_row.get('input_text'):
    whisperx_loaded_audio = whisperx.load_audio(media_path)
  
  # Transcribe audio to text (whisperX) if text isn't provided
  if not input_row.get('input_text'):
    input_row['input_text'] = ''
    sys.stdout.write('\n  Transcribing audio to text\n')
    sys.stdout.write('\nWHISPERX: model.transcribe\n')
    generated_transcript_data = whisperx_model('small').transcribe(whisperx_loaded_audio, language=lang, batch_size=4)
    for segment in generated_transcript_data['segments']:
      input_row['input_text'] += segment.get('text', '') + ' '
    sys.stdout.write('END\n\n')
    sys.stdout.write('\n')
  
  # Load transcript
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
  
  # Align transcript to audio
  sys.stdout.write('  Aligning text to audio ({0})\n'.format(options.alignment_method))
  segments_with_word_data = []
  if options.alignment_method == 'whisperx':
    rough_segments = [{
      'text': modified_transcript,
      'start': 0,
      'end': librosa.get_duration(path=media_path),
    }]
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(rough_segments, align_model, metadata, whisperx_loaded_audio, device, return_char_alignments=False)
    segments_with_word_data = result['segments']
  elif options.alignment_method == 'stable_ts':
    result = stable_ts_model('small').align(media_path, modified_transcript, language=lang, vad=True)
    words = []
    for word in result.all_words_or_segments():
      word_dict = word.to_dict()
      word_dict['word'] = word_dict['word'].strip()
      words.append(word_dict)
    segments_with_word_data = [{
      'words': words
    }]
  
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
      
      if 'start' in word_data and 'end' in word_data and word_data['end'] > word_data['start']:
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
        formatted_start = seconds_to_vtt_timestamp(block['start_seconds'], options.time_offset)
        formatted_end = seconds_to_vtt_timestamp(block['end_seconds'], options.time_offset)
        vtt += textwrap.dedent(f'''\
          
          {{"block": {b + 1}}}
          {formatted_start} --> {formatted_end}
        ''')
        for p, phrase in enumerate(block['phrases']):
          for w, word in enumerate(phrase['words']):
            if options.word_timestamps:
              formatted_word_start = seconds_to_vtt_timestamp(word['start_seconds'], options.time_offset)
              vtt += f'<{formatted_word_start}>'
            vtt += word['text'] + ' '
          vtt = vtt.strip() + '\n'
        
    # Phrase-based VTT
    if vtt_type == 'phrase':
      vtt = vtt_phrase
      for b, block in enumerate(blocks):
        for p, phrase in enumerate(block['phrases']):
          formatted_start = seconds_to_vtt_timestamp(phrase['start_seconds'], options.time_offset)
          formatted_end = seconds_to_vtt_timestamp(phrase['end_seconds'], options.time_offset)
          vtt += textwrap.dedent(f'''\
      
            {{"block": {b + 1}, "phrase": {p + 1}}}
            {formatted_start} --> {formatted_end}
          ''')
          for w, word in enumerate(phrase['words']):
            if options.word_timestamps:
              formatted_word_start = seconds_to_vtt_timestamp(word['start_seconds'], options.time_offset)
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
            formatted_start = seconds_to_vtt_timestamp(word['start_seconds'], options.time_offset)
            formatted_end = seconds_to_vtt_timestamp(word['end_seconds'], options.time_offset)
            vtt += textwrap.dedent(f'''\
      
            {{"block": {b + 1}, "phrase": {p + 1}, "word": {w + 1}}}
              {formatted_start} --> {formatted_end}
            ''')
            if options.word_timestamps:
              formatted_word_start = seconds_to_vtt_timestamp(word['start_seconds'], options.time_offset)
              vtt += f'<{formatted_word_start}>'
            vtt += word['text'] + '\n'

    vtt_output_folder = os.path.join(output_folder, vtt_type)
    if not os.path.exists(vtt_output_folder):
      os.makedirs(vtt_output_folder)
    vtt_output_path = os.path.join(vtt_output_folder, '{0}.vtt'.format(input_row.get('id')))
    with open(vtt_output_path, 'w', encoding='utf-8') as f:
      f.write(vtt)
    
sys.stdout.write('\nDone!\n\n')
