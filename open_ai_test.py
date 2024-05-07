import os
import wave
import pyaudio
from openai import OpenAI
from pydub import AudioSegment
import io

NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def transcribe_chunk(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            # prompt="english or bahasa please"
            # language="en"
        )
    return transcription.text

def record_chunk(p, stream, chunk_length=5, output_path="temp.wav"):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    # Combine the frames into a single byte object
    audio_data = b''.join(frames)

    # Convert audio data to AudioSegment object
    audio_segment = AudioSegment(
        audio_data,
        sample_width=p.get_sample_size(pyaudio.paInt16),
        frame_rate=16000,
        channels=1
    )

    # Export the audio segment to a WAV file
    audio_segment.export(output_path, format="wav")

    # Return the path to the WAV audio file
    return output_path


def main():
    # Initialize OpenAI client
    client = OpenAI()
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = ""

    try:
        while True:
            audio_path = record_chunk(p, stream)
            transcription = transcribe_chunk(client, audio_path)
            print(NEON_GREEN + transcription + RESET_COLOR)
            # Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "
    except KeyboardInterrupt:
        print("Stopping...")
        # Write the accumulated transcription to the Log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG:" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    main()
