# Kokoro-82M TTS 모델 Python 사용 가이드

# 1. 설치
"""
필수 의존성 설치:
pip install kokoro>=0.9.4 soundfile
apt-get install espeak-ng  # Linux
brew install espeak        # macOS

Windows의 경우:
- espeak-ng Windows 바이너리 다운로드 및 설치
- Microsoft Visual Studio C++ Community Edition 설치 (컴파일러 필요)
- NVIDIA CUDA Toolkit 설치 (GPU 사용 시)
"""

# 2. 기본 사용법
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

def basic_tts_usage():
    """기본 TTS 사용법"""
    
    # 파이프라인 초기화
    # 🇺🇸 'a' => American English
    # 🇬🇧 'b' => British English  
    # 🇯🇵 'j' => Japanese (pip install misaki[ja] 필요)
    # 🇨🇳 'z' => Mandarin Chinese (pip install misaki[zh] 필요)
    pipeline = KPipeline(lang_code='a')
    
    text = '''
    안녕하세요! Kokoro는 82백만 개의 파라미터를 가진 오픈 웨이트 TTS 모델입니다.
    가벼운 아키텍처에도 불구하고 더 큰 모델들과 비교할 수 있는 품질을 제공합니다.
    '''
    
    # 음성 생성 (기본 음성 사용)
    generator = pipeline(text)
    
    for i, (gs, ps, audio) in enumerate(generator):
        print(f"청크 {i}: gs={gs}, ps={ps}")
        
        # Jupyter에서 재생
        display(Audio(data=audio, rate=24000, autoplay=i==0))
        
        # 파일로 저장
        sf.write(f'output_{i}.wav', audio, 24000)
        
        break  # 첫 번째 청크만 처리

def voice_selection_usage():
    """다양한 음성 사용법"""
    
    pipeline = KPipeline(lang_code='a')
    
    # 사용 가능한 음성들
    available_voices = [
        'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 
        'af_jessica', 'af_kore', 'af_nicole', 'af_nova',
        'af_river', 'af_sarah', 'af_sky', 'am_adam',
        'am_apollo', 'am_daniel', 'am_eric', 'am_michael'
    ]
    
    text = "안녕하세요, 저는 Kokoro TTS 모델입니다."
    
    # 특정 음성으로 생성
    for voice in available_voices[:3]:  # 처음 3개 음성만 테스트
        print(f"\n=== {voice} 음성으로 생성 ===")
        
        generator = pipeline(text, voice=voice)
        
        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(f'{voice}_output.wav', audio, 24000)
            print(f"{voice} 음성 파일 저장 완료")
            break

def advanced_usage():
    """고급 사용법 - 음성 블렌딩"""
    
    pipeline = KPipeline(lang_code='a')
    
    text = "이것은 음성 블렌딩 테스트입니다."
    
    # 여러 음성 조합 (비율 지정)
    # 형식: "voice1:weight1,voice2:weight2"
    mixed_voice = "af_sarah:60,am_adam:40"  # 60% Sarah + 40% Adam
    
    generator = pipeline(text, voice=mixed_voice)
    
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write('mixed_voice_output.wav', audio, 24000)
        print("음성 블렌딩 파일 저장 완료")
        break

def batch_processing():
    """배치 처리 예제"""
    
    pipeline = KPipeline(lang_code='a')
    
    texts = [
        "첫 번째 텍스트입니다.",
        "두 번째 텍스트입니다.", 
        "세 번째 텍스트입니다."
    ]
    
    voices = ['af_bella', 'af_sarah', 'am_adam']
    
    for i, (text, voice) in enumerate(zip(texts, voices)):
        print(f"\n처리 중: {i+1}/{len(texts)}")
        
        generator = pipeline(text, voice=voice)
        
        for j, (gs, ps, audio) in enumerate(generator):
            filename = f'batch_output_{i+1}_{voice}.wav'
            sf.write(filename, audio, 24000)
            print(f"저장 완료: {filename}")
            break

def streaming_synthesis():
    """스트리밍 음성 합성"""
    
    pipeline = KPipeline(lang_code='a')
    
    long_text = """
    이것은 긴 텍스트의 예제입니다. Kokoro TTS는 긴 텍스트를 
    자동으로 청크로 나누어 처리합니다. 각 청크는 독립적으로 
    처리되어 메모리 효율성을 높입니다. 이는 특히 긴 문서나 
    책을 음성으로 변환할 때 유용합니다.
    """
    
    generator = pipeline(long_text, voice='af_bella')
    
    audio_chunks = []
    
    for i, (gs, ps, audio) in enumerate(generator):
        print(f"청크 {i+1} 처리 완료 (길이: {len(audio)} 샘플)")
        audio_chunks.append(audio)
        
        # 각 청크를 개별 파일로 저장
        sf.write(f'chunk_{i+1}.wav', audio, 24000)
    
    # 모든 청크를 하나의 파일로 합치기
    if audio_chunks:
        import numpy as np
        combined_audio = np.concatenate(audio_chunks)
        sf.write('combined_output.wav', combined_audio, 24000)
        print(f"전체 {len(audio_chunks)}개 청크를 combined_output.wav로 저장")

def multilingual_usage():
    """다국어 지원"""
    
    # 영어
    en_pipeline = KPipeline(lang_code='a')
    en_text = "Hello, this is English text."
    
    generator = en_pipeline(en_text, voice='af_bella')
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write('english_output.wav', audio, 24000)
        break
    
    # 일본어 (misaki[ja] 설치 필요)
    try:
        ja_pipeline = KPipeline(lang_code='j')
        ja_text = "こんにちは、これは日本語のテストです。"
        
        generator = ja_pipeline(ja_text)
        for i, (gs, ps, audio) in enumerate(generator):
            sf.write('japanese_output.wav', audio, 24000)
            break
    except Exception as e:
        print(f"일본어 지원을 위해 'pip install misaki[ja]' 실행 필요: {e}")

def file_processing():
    """파일에서 텍스트 읽어서 처리"""
    
    pipeline = KPipeline(lang_code='a')
    
    # 텍스트 파일 읽기 예제
    sample_text = """
    이것은 파일에서 읽은 텍스트입니다.
    여러 줄로 구성되어 있습니다.
    Kokoro TTS가 이를 음성으로 변환합니다.
    """
    
    # 임시 파일 생성
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # 파일에서 읽어서 처리
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    generator = pipeline(text, voice='af_sarah')
    
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write('file_output.wav', audio, 24000)
        print("파일 처리 완료: file_output.wav")
        break

def get_model_info():
    """모델 정보 확인"""
    
    pipeline = KPipeline(lang_code='a')
    
    print("=== Kokoro-82M 모델 정보 ===")
    print("- 파라미터: 82M")
    print("- 라이선스: Apache 2.0") 
    print("- 샘플링 레이트: 24,000 Hz")
    print("- 지원 언어: 영어, 일본어, 중국어")
    print("- 아키텍처: StyleTTS 2 + ISTFTNet (decoder-only)")
    
    # 사용 가능한 음성 목록
    voices = [
        'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica',
        'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 
        'af_sky', 'am_adam', 'am_apollo', 'am_daniel', 'am_eric', 'am_michael'
    ]
    
    print(f"\n사용 가능한 음성 ({len(voices)}개):")
    for voice in voices:
        print(f"  - {voice}")

# 사용 예제 실행
if __name__ == "__main__":
    print("=== Kokoro-82M TTS 사용 예제 ===\n")
    
    # 각 함수를 차례로 실행
    try:
        print("1. 기본 사용법")
        basic_tts_usage()
        
        print("\n2. 음성 선택")
        voice_selection_usage()
        
        print("\n3. 고급 사용법 (음성 블렌딩)")
        advanced_usage()
        
        print("\n4. 배치 처리")
        batch_processing()
        
        print("\n5. 스트리밍 합성")
        streaming_synthesis()
        
        print("\n6. 다국어 지원")
        multilingual_usage()
        
        print("\n7. 파일 처리")
        file_processing()
        
        print("\n8. 모델 정보")
        get_model_info()
        
        print("\n모든 예제 실행 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("먼저 필요한 패키지들을 설치해주세요:")
        print("pip install kokoro>=0.9.4 soundfile")
        print("apt-get install espeak-ng  # Linux")