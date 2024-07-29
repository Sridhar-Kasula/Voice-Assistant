"""
AI Voice Assistant using OpenAI API
By Sridhar Kasula
"""

# Installations
#!pip install --quiet soundfile sounddevice openai


# Imports
import sounddevice
import soundfile
from scipy.io.wavfile import write
import openai
import getpass


# Initilizing OpenAI client
key = getpass.getpass("Enter OpenAI key")
client = openai.OpenAI(
  api_key= key #os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)


# Initializing system_message to store query and response history

#  Below is an example of system message
"""
I am a hiking enthusiast named Forest who helps people discover hikes in their area.
If no area is specified, I will default to near Rainier National Park.
I will then provide three suggestions for nearby hikes that vary in length.
I will also share an interesting fact about the local nature on the hikes when making a recommendation.
"""

# Create a system message
system_message = input("Please enter system message for the model, I will provide answers around this message") 
if len(system_message) == 0:
    system_message = "I am an AI assistant, I am here to help you answer genral question"

# Initialize messages array (to consider previous prompts and responses for future responses)
messages_array = [{"role": "system", "content": system_message}]


# Voice input
def listen():
    fs= 44100
    second =  int(4)
    print("\nRecording.....\n")
    record_voice = sounddevice.rec( int ( second * fs ) , samplerate = fs , channels = 2 )
    sounddevice.wait()
    
    write("out.wav", fs, record_voice)
    print("Finished Recording...\n")
    
    #Audio-to-Text using GPT model
    audio_file= open("out.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model = "whisper-1", 
        file = audio_file
    )
    
    # Append query to message_array to store the history
    messages_array.append({"role": "user", "content": transcription.text})
    print("Query: " + transcription.text + "\n")
    return transcription.text


# Get response from OpenAI
def get_response(query):
    response = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    temperature = 0.7,
    max_tokens = 400,
    messages = messages_array)
    
    generated_text = response.choices[0].message.content

    # Add generated text to messages array
    messages_array.append({"role": "assistant", "content": generated_text})
    
    # Text to Audio
    audio_response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input = generated_text)
    
    audio_response.stream_to_file("output.mp3")
    
    # Print the response
    print("Response: " + generated_text + "\n")
    return audio_response


# Below is the full assembly
while True:
    # Get confirmation to continue
    do_you_want_to_continue = input("Enter to continue (or type 'quit' to exit): ")
    
    if do_you_want_to_continue.lower() == "quit":
        break
    if len(do_you_want_to_continue) == 0:
        input_text = listen()
        get_response(input_text)
        file, fs = soundfile.read("output.mp3")
        sounddevice.play(file, fs)
        sounddevice.wait()
        continue

