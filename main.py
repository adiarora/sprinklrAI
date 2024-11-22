import whisper
import os
import warnings
from transformers import pipeline, logging
import textwrap
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Suppress the FP16 warnings
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU")

# Suppress other warnings
warnings.filterwarnings("ignore")

# Set logging level for transformers to ERROR to suppress info and warnings
logging.set_verbosity_error()

# Set TOKENIZERS_PARALLELISM environment variable to false to suppress parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Whisper model
model = whisper.load_model("base")

# Load summarization pipeline
summarizer = pipeline("summarization")

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize dictionary to store transcriptions and summaries
transcriptions = {}

def split_text(text, max_length):
    """Splits text into chunks of max_length."""
    sentences = text.split('. ')
    chunks = []
    chunk = []
    length = 0

    for sentence in sentences:
        if length + len(sentence) + 1 > max_length:
            chunks.append('. '.join(chunk) + '.')
            chunk = []
            length = 0
        chunk.append(sentence)
        length += len(sentence) + 1

    if chunk:
        chunks.append('. '.join(chunk) + '.')

    return chunks

def format_paragraph(text, width=80):
    """Formats text into paragraphs with specified width."""
    return "\n".join(textwrap.wrap(text, width))

while True:
    # Ask user for name of audio file
    audio_file = input("Enter the name of an audio file (or 'done' to finish): ")
    
    if audio_file.lower() == "done":
        break
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print("This file does not exist. Please try again:")
        continue
    
    # Transcribe audio file
    result = model.transcribe(audio_file)
    transcription_text = result["text"]
    
    # Split text if necessary and summarize each chunk
    max_chunk_length = 1024  # This length is typically safe for most summarization models
    chunks = split_text(transcription_text, max_chunk_length)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    # Combine the summaries into one summary text
    full_summary = ' '.join(summaries)
    
    # Store transcription and summary in dictionary
    transcriptions[audio_file] = {
        "transcription": transcription_text,
        "summary": full_summary
    }

# Write all transcriptions and summaries to a file
with open("all_transcriptions.txt", "w") as f:
    for filename, content in transcriptions.items():
        f.write(f"{filename}:\n")
        f.write("Transcription:\n")
        f.write(format_paragraph(content['transcription']) + "\n\n")
        f.write("Summary:\n")
        f.write(format_paragraph(content['summary']) + "\n\n")

# Combine all summaries into one text
all_summaries = " ".join(content["summary"] for content in transcriptions.values())

# Split all summaries text into chunks if necessary and generate a comprehensive insight summary
max_chunk_length = 1024  # This length is typically safe for most summarization models
chunks = split_text(all_summaries, max_chunk_length)
final_summaries = []

for chunk in chunks:
    summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    final_summaries.append(summary)

# Combine the final summaries into one insight summary
insight_summary = ' '.join(final_summaries)

# Further summarize the insight summary to shorten it to a medium length
shortened_insight_summary = summarizer(insight_summary, max_length=400, min_length=200, do_sample=False)[0]['summary_text']

# Perform sentiment analysis on the shortened insight summary
sentences = nltk.sent_tokenize(shortened_insight_summary)
positive_points = []
neutral_points = []
negative_points = []

for sentence in sentences:
    sentiment_score = sentiment_analyzer.polarity_scores(sentence)
    if sentiment_score['compound'] > 0.3:
        positive_points.append(sentence)
    elif sentiment_score['compound'] < -0.3:
        negative_points.append(sentence)
    else:
        neutral_points.append(sentence)

# Write the insight summary and sentiment analysis to a file
with open("insight_summary.txt", "w") as f:
    f.write("Insight Summary:\n")
    f.write(format_paragraph(shortened_insight_summary) + "\n\n")
    
    f.write("Sentiment Analysis:\n")
    
    f.write("Positive Points:\n")
    f.write(format_paragraph(' '.join(positive_points)) + "\n\n")
    
    f.write("Neutral Points:\n")
    f.write(format_paragraph(' '.join(neutral_points)) + "\n\n")
    
    f.write("Negative Points:\n")
    f.write(format_paragraph(' '.join(negative_points)) + "\n\n")

print("Transcriptions and summaries completed and saved to 'all_transcriptions.txt'.")
print("Insight summary and sentiment analysis completed and saved to 'insight_summary.txt'.")
