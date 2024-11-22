# SprinklrAI

## Project Overview

**SprinklrAI** is a research project designed to automate the transcription and summarization of customer review videos. By leveraging OpenAI's Whisper for transcription and advanced NLP pipelines, this tool delivers summarized insights and sentiment analysis to help identify customer feedback trends, enhancing product review insights and decision-making.

## Features
1. **Audio Transcription**: Converts audio files into highly accurate text using OpenAI's Whisper model.
2. **Summarization**: Generates concise summaries from transcriptions using Transformer-based models.
3. **Sentiment Analysis**: Extracts positive, neutral, and negative sentiment points using VADER.
4. **Insight Summary**: Combines multiple summaries into a comprehensive report.

## Requirements

To run the program, you need:

1. Python 3.8 or higher
2. The following Python libraries:
   - `openai-whisper`
   - `transformers`
   - `nltk`
3. Run 'python main.py'
4. Use titles of downloaded audio files for input to program