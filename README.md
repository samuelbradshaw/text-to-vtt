# Text to VTT

Text to VTT is a Python script for generating [WebVTT](https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API) caption files with block, phrase, and/or word-level timestamps, given an audio or video recording and an existing transcript.

This script relies on free and open-source machine learning libraries, including [OpenAI Whisper](https://github.com/openai/whisper) (speech recognition), [WhisperX](https://github.com/m-bain/whisperX) (word-level forced alignment), and [Demucs](https://github.com/facebookresearch/demucs) (audio separation).

Example WebVTT output files can be found in the `sample` folder in this repository.


## Installation

1. Verify that Python 3 is installed. You will also need a package manager such as Homebrew (macOS) or apt (Linux).

2. Clone or download Text to VTT.

3. In Terminal, go to the `text-to-vtt` directory:

```
cd /path/to/text-to-vtt
```

4. Install dependencies:

```
brew install ffmpeg
pip3 install -r requirements.txt
```


## Running the script

1. Create a CSV or TSV file with required input data (more details below).

2. Run the script, passing in the input file.
```
python3 text-to-vtt.py [path-to-csv-or-tsv]
```

Text to VTT will create a work files folder (under the same directory as the script) for downloading and processing content (`_workfiles`), and an output folder for the generated WebVTT files (`_output`).

### Input file

Text to VTT takes a CSV or TSV spreadsheet file as input. The spreadsheet should have a row for each audio or video file with the following columns:

* **id** (required) – A unique identifier for the media file, such as a number or slug. This will be used as the output filename and will appear in the WebVTT.

* **title** (required) – Title for the media file. This will appear in the WebVTT.

* **lang** (required) – BCP 47 language tag representing the language of the media file. This will be used to load the correct language model and will appear in the WebVTT. These are the current supported language tags: ['de', 'en', 'es', 'fr', 'it', 'ja', 'nl', 'pt', 'uk', 'zh'].

* **input_media** (required if file not present) – URL to download the audio or video file in a standard format such as MP3, WAV, or MP4. Alternatively, if you already have the file locally, you can put the file under `_workfiles/media/` with a filename that matches the `id` specified above. If a matching file is present, this column will be ignored.

* **input_text** (recommended if file not present) – Transcript text. Alternatively, you can put the transcript in a text file (extension .txt) under `_workfiles/text/` with a filename that matches the `id` specified above. If a matching file is present, this column will be ignored. If no text is provided, Text to VTT will generate a transcript from the audio. For best results, plain Unicode text (not HTML or Markdown) is recommended, with double line breaks between paragraph blocks. Words inside &lt;angle brackets&gt; or [square brackets] will be ignored.

* **use_voice_isolation** (optional) – Whether the audio should be processed for voice isolation. This is recommended for music, but isn’t needed for clear spoken audio. Default: False. Supported values: True (or Yes or 1) or False (or No or 0).

* **vtt_output_types** (optional) – Whether each cue in the WebVTT should include a full block of the transcript (split by double line breaks), a phrase, or a single word. Multiple values will output multiple WebVTT files. Default: ['phrase']. Supported values: One or more of the following: 'block', 'phrase', 'word'.

* **break_phrases_at** (optional) – Instruction that determines where blocks should be split to form phrases. For lyrics or for a transcript that’s been manually broken into phrases, single_line_breaks is recommended. For a paragraph-based transcript, sentence_punctuation and character_count are recommended. Default: ['single_line_breaks']. Supported values: One or more of the following: 'single_line_breaks', 'sentence_punctuation', 'character_count'.

* **max_phrase_character_count** (optional) – Maximum number of characters in a phrase. Only applies when `break_phrases_at` includes character_count. Default: 40. Supported values: Integer between 1 and 500.

* **word_timestamps** (optional) – Whether [timestamp tags](https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API#cue_payload_text_tags) should be added before each word in the WebVTT cues. This is useful for karaoke and other follow-along applications, but may not be supported in all contexts that use WebVTT. Default: False. Supported values: True (or Yes or 1) or False (or No or 0).

* **time_offset** (optional) – Number of seconds that should be added or removed from each calculated timestamp. Default: -0.3. Supported values: Any number (including decimals) between -5.0 and 5.0).

An example input file can be found in the `sample` folder in this repository.
