import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement, EncoderASR

def enhance_and_transcribe(audio_path):
    # Load the speech enhancement model
    enhancement_model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir="models/enhancement")

    # Load the ASR model
    asr_model = EncoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="models/asr")

    # Load the noisy speech signal
    noisy_speech, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Sample rate needs to be 16 kHz."

    # Enhance the speech
    enhanced_speech = enhancement_model.enhance_batch(noisy_speech.unsqueeze(0))

    # Transcribe the enhanced speech
    transcription = asr_model.transcribe_batch(enhanced_speech)

    return transcription

# Example usage
audio_path = 'audio/101-88e1fd5a-7e09-467d-a701-fdbf3a82a886 (Wed Dec  8 15_21_44 2021)_redu.wav'
transcription = enhance_and_transcribe(audio_path)
print("Transcription:", transcription)
