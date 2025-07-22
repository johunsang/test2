# Kokoro 모델 수동 설치 및 사용

import os
import torch
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download
import json

def download_kokoro_model():
    """Kokoro 모델과 설정 파일 다운로드"""
    
    model_repo = "hexgrad/Kokoro-82M"
    
    # 모델 파일들 다운로드
    files_to_download = [
        "kokoro-v1_0.pth",  # 모델 가중치
        "config.json",      # 설정 파일
    ]
    
    downloaded_files = {}
    
    for filename in files_to_download:
        try:
            print(f"다운로드 중: {filename}")
            file_path = hf_hub_download(
                repo_id=model_repo,
                filename=filename,
                cache_dir="./model_cache"
            )
            downloaded_files[filename] = file_path
            print(f"✅ {filename} 다운로드 완료: {file_path}")
        except Exception as e:
            print(f"❌ {filename} 다운로드 실패: {e}")
    
    return downloaded_files

def download_voice_files():
    """음성 파일들 다운로드"""
    
    model_repo = "hexgrad/Kokoro-82M"
    
    # 일부 음성 파일들
    voice_files = [
        "voices/af_bella.pt",
        "voices/af_sarah.pt", 
        "voices/af_heart.pt",
        "voices/am_adam.pt"
    ]
    
    downloaded_voices = {}
    
    for voice_file in voice_files:
        try:
            print(f"음성 파일 다운로드 중: {voice_file}")
            file_path = hf_hub_download(
                repo_id=model_repo,
                filename=voice_file,
                cache_dir="./model_cache"
            )
            voice_name = os.path.basename(voice_file).replace('.pt', '')
            downloaded_voices[voice_name] = file_path
            print(f"✅ {voice_name} 다운로드 완료")
        except Exception as e:
            print(f"❌ {voice_file} 다운로드 실패: {e}")
    
    return downloaded_voices

class SimpleKokoroTTS:
    """간단한 Kokoro TTS 구현"""
    
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            # 설정 파일 로드
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print("✅ 설정 파일 로드 완료")
            
            # 모델 로드 (PyTorch)
            self.model = torch.load(self.model_path, map_location='cpu')
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
    
    def synthesize(self, text, voice_path=None):
        """텍스트를 음성으로 변환 (기본 구현)"""
        print(f"텍스트 변환 중: {text}")
        
        # 실제 TTS 구현은 복잡하므로, 여기서는 더미 오디오 생성
        # 실제로는 모델을 통해 음성을 생성해야 함
        sample_rate = 24000
        duration = len(text) * 0.1  # 텍스트 길이에 비례한 duration
        samples = int(sample_rate * duration)
        
        # 더미 오디오 (실제 구현에서는 모델 추론 결과)
        audio = np.random.randn(samples) * 0.1
        
        return audio, sample_rate

def alternative_setup():
    """대안적인 설정 방법"""
    
    print("=== Kokoro 대안 설정 ===\n")
    
    # 1. 모델 파일 다운로드
    print("1. 모델 파일 다운로드")
    model_files = download_kokoro_model()
    
    if not model_files:
        print("모델 다운로드 실패. 수동으로 다운로드해야 합니다.")
        return
    
    # 2. 음성 파일 다운로드
    print("\n2. 음성 파일 다운로드")
    voice_files = download_voice_files()
    
    # 3. 간단한 TTS 클래스 사용
    if "kokoro-v1_0.pth" in model_files and "config.json" in model_files:
        print("\n3. TTS 초기화")
        tts = SimpleKokoroTTS(
            model_files["kokoro-v1_0.pth"],
            model_files["config.json"]
        )
        
        # 4. 테스트 음성 생성
        print("\n4. 테스트 음성 생성")
        text = "Hello, this is a test of Kokoro TTS."
        audio, sr = tts.synthesize(text)
        
        # 5. 오디오 저장
        output_file = "test_output.wav"
        sf.write(output_file, audio, sr)
        print(f"✅ 테스트 오디오 저장: {output_file}")

def manual_install_guide():
    """수동 설치 가이드"""
    
    print("=== 수동 설치 가이드 ===\n")
    
    print("1. 필요한 패키지 설치:")
    print("   pip install torch torchaudio transformers soundfile huggingface_hub")
    
    print("\n2. Hugging Face에서 수동 다운로드:")
    print("   - https://huggingface.co/hexgrad/Kokoro-82M/tree/main")
    print("   - kokoro-v1_0.pth 다운로드")
    print("   - config.json 다운로드") 
    print("   - voices/ 폴더의 .pt 파일들 다운로드")
    
    print("\n3. espeak 설치:")
    print("   - macOS: brew install espeak")
    print("   - Ubuntu: sudo apt-get install espeak-ng")
    
    print("\n4. 대안적인 TTS 라이브러리:")
    print("   - pip install TTS  # Coqui TTS")
    print("   - pip install pyttsx3  # 시스템 TTS")
    print("   - pip install gTTS  # Google TTS")

def test_alternative_tts():
    """대안 TTS 라이브러리 테스트"""
    
    print("=== 대안 TTS 테스트 ===\n")
    
    # 1. pyttsx3 테스트 (시스템 TTS)
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        
        text = "안녕하세요, 이것은 시스템 TTS 테스트입니다."
        engine.save_to_file(text, 'pyttsx3_output.wav')
        engine.runAndWait()
        
        print("✅ pyttsx3 TTS 테스트 성공")
        
    except ImportError:
        print("❌ pyttsx3 없음. 설치: pip install pyttsx3")
    except Exception as e:
        print(f"❌ pyttsx3 오류: {e}")
    
    # 2. gTTS 테스트 (Google TTS)
    try:
        from gtts import gTTS
        import io
        
        text = "Hello, this is Google TTS test."
        tts = gTTS(text=text, lang='en')
        
        # 메모리에서 처리
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # 파일로 저장
        with open('gtts_output.mp3', 'wb') as f:
            f.write(fp.getvalue())
        
        print("✅ gTTS 테스트 성공: gtts_output.mp3")
        
    except ImportError:
        print("❌ gTTS 없음. 설치: pip install gTTS")
    except Exception as e:
        print(f"❌ gTTS 오류: {e}")

# 실행 예제
if __name__ == "__main__":
    print("Kokoro 설치 문제 해결\n")
    
    # 필요한 패키지 설치 확인
    required_packages = ['torch', 'soundfile', 'huggingface_hub', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 설치됨")
        except ImportError:
            print(f"❌ {package} 없음. 설치 필요: pip install {package}")
    
    print("\n" + "="*50)
    
    # 대안 방법들 실행
    try:
        print("\n대안 1: 수동 모델 다운로드")
        alternative_setup()
        
    except Exception as e:
        print(f"대안 1 실패: {e}")
        
        print("\n대안 2: 다른 TTS 라이브러리 사용")
        test_alternative_tts()
        
        print("\n대안 3: 수동 설치 가이드")
        manual_install_guide()