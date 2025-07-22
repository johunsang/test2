#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 Bllossom 모델 텍스트 생성 예제
간단하고 실용적인 텍스트 생성 데모
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class SimpleBlossomChat:
    def __init__(self, model_name=None):
        """간단한 한국어 텍스트 생성기 초기화"""
        # 여러 모델 옵션 제공
        self.available_models = {
            "bllossom": "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B",
            "bllossom_text": "Bllossom/llama-3.2-Korean-Bllossom-3B",  # 텍스트 전용
            "eeve": "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",  # 대안 한국어 모델
            "solar": "upstage/SOLAR-10.7B-Instruct-v1.0"  # 더 가벼운 대안
        }
        
        # 모델 선택
        if model_name is None:
            model_name = self.select_model()
        
        self.model_name = self.available_models.get(model_name, model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🌸 모델 로딩 중: {self.model_name}")
        print(f"📱 디바이스: {self.device}")
        
        self.load_model()
    
    def select_model(self):
        """사용할 모델 선택"""
        print("\n🤖 사용할 모델을 선택하세요:")
        print("1. bllossom - Bllossom Vision (비전 기능 포함, 큰 모델)")
        print("2. bllossom_text - Bllossom 텍스트 전용 (3B, 가벼움)")  
        print("3. eeve - EEVE 한국어 모델 (안정적)")
        print("4. solar - SOLAR 모델 (빠름)")
        
        choice = input("선택 (1-4, 기본값: 2): ").strip()
        
        model_map = {"1": "bllossom", "2": "bllossom_text", "3": "eeve", "4": "solar"}
        return model_map.get(choice, "bllossom_text")
    
    def load_model(self):
        """모델과 토크나이저 로드 (에러 복구 기능 포함)"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"📥 시도 {attempt + 1}/{max_retries}: 토크나이저 로딩...")
                
                # 토크나이저 로드
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False  # 호환성을 위해
                )
                
                # 패딩 토큰 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"📥 모델 로딩 중...")
                
                # 기본 모델 로드 설정 (안전한 설정)
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True
                }
                
                # 단계별 로딩 시도
                if torch.cuda.is_available():
                    print("🚀 GPU 감지됨. GPU 최적화 시도...")
                    try:
                        # 먼저 간단한 GPU 로딩 시도
                        model_kwargs["device_map"] = "auto"
                        
                        # 메모리가 충분하지 않으면 양자화 시도
                        if attempt > 0:  # 첫 번째 시도 실패시
                            try:
                                from transformers import BitsAndBytesConfig
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True  # 4bit 대신 8bit 사용 (더 안정적)
                                )
                                model_kwargs["quantization_config"] = quantization_config
                                print("🔧 8bit 양자화 적용")
                            except ImportError:
                                print("⚠️ bitsandbytes 없음. CPU로 로딩 시도")
                                model_kwargs = {
                                    "trust_remote_code": True,
                                    "torch_dtype": torch.float32
                                }
                    except Exception as gpu_error:
                        print(f"⚠️ GPU 로딩 실패: {gpu_error}")
                        print("🔄 CPU 로딩으로 전환...")
                        model_kwargs = {
                            "trust_remote_code": True,
                            "torch_dtype": torch.float32
                        }
                
                # 모델 로드
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # CPU로 이동 (필요시)
                if not torch.cuda.is_available():
                    self.model = self.model.to("cpu")
                
                print("✅ 모델 로딩 완료!")
                self.print_model_info()
                return
                
            except Exception as e:
                print(f"❌ 로딩 실패 (시도 {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    print("🔄 다른 설정으로 재시도...")
                    # 다음 시도를 위해 더 안전한 설정 사용
                    if "bllossom" in self.model_name.lower():
                        # Bllossom 모델이 실패하면 더 간단한 모델로 전환
                        self.model_name = self.available_models["solar"]
                        print(f"🔄 대안 모델로 전환: {self.model_name}")
                else:
                    print("❌ 모든 시도 실패")
                    print("💡 해결책:")
                    print("  1. 인터넷 연결 확인")
                    print("  2. pip install transformers --upgrade")
                    print("  3. 더 작은 모델 사용")
                    raise
    
    def print_model_info(self):
        """모델 정보 출력"""
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"📊 파라미터 수: {param_count:,}")
            print(f"💾 디바이스: {next(self.model.parameters()).device}")
            print(f"📝 모델: {self.model_name.split('/')[-1]}")
        except:
            print("📊 모델 정보를 가져올 수 없습니다.")
    
    def generate_text(self, prompt, max_length=300, temperature=0.7):
        """텍스트 생성 함수 (모델별 최적화)"""
        try:
            # 모델에 따른 프롬프트 포맷팅
            if "bllossom" in self.model_name.lower():
                # Bllossom 모델용 포맷
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif "eeve" in self.model_name.lower():
                # EEVE 모델용 포맷
                formatted_prompt = f"### 질문: {prompt}\n\n### 답변:"
            elif "solar" in self.model_name.lower():
                # SOLAR 모델용 포맷
                formatted_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
            else:
                # 기본 포맷
                formatted_prompt = f"질문: {prompt}\n답변:"
            
            # 토크나이징 (에러 처리 포함)
            try:
                inputs = self.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False
                )
            except Exception as tokenize_error:
                print(f"⚠️ 토크나이징 오류: {tokenize_error}")
                # 간단한 포맷으로 재시도
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # 디바이스로 이동
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 생성 파라미터 설정 (안전한 값들)
            generation_config = {
                "max_length": len(inputs['input_ids'][0]) + max_length,
                "min_length": len(inputs['input_ids'][0]) + 10,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,
                "early_stopping": True
            }
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거 및 정리
            if "assistant<|end_header_id|>" in response:
                response = response.split("assistant<|end_header_id|>")[-1].strip()
            elif "### 답변:" in response:
                response = response.split("### 답변:")[-1].strip()
            elif "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()
            elif "답변:" in response:
                response = response.split("답변:")[-1].strip()
            elif formatted_prompt in response:
                response = response.replace(formatted_prompt, "").strip()
            
            # 빈 응답 방지
            if not response.strip():
                response = "죄송합니다. 응답을 생성할 수 없습니다. 다른 질문을 시도해보세요."
            
            return response
            
        except Exception as e:
            error_msg = f"텍스트 생성 중 오류가 발생했습니다: {e}"
            print(f"❌ {error_msg}")
            return error_msg

def main():
    """메인 실행 함수"""
    print("🌸 한국어 Bllossom 텍스트 생성 데모")
    print("=" * 40)
    
    try:
        # 모델 초기화
        chat_bot = SimpleBlossomChat()
        
        # 샘플 프롬프트들
        sample_prompts = [
            "파이썬 프로그래밍의 장점에 대해 설명해주세요.",
            "한국의 전통 음식에 대해 소개해주세요.",
            "인공지능의 미래에 대한 당신의 생각은?",
            "효과적인 학습 방법을 알려주세요.",
            "환경 보호를 위해 개인이 할 수 있는 일들은?"
        ]
        
        print("\n🎯 사용 방법을 선택하세요:")
        print("1. 직접 질문 입력")
        print("2. 샘플 프롬프트 사용")
        print("3. 대화형 채팅")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            # 직접 입력
            user_prompt = input("\n질문을 입력하세요: ")
            print("\n🤖 생성 중...")
            
            response = chat_bot.generate_text(user_prompt)
            print(f"\n🌸 Bllossom:\n{response}")
            
        elif choice == "2":
            # 샘플 프롬프트
            print("\n📝 샘플 프롬프트들:")
            for i, prompt in enumerate(sample_prompts, 1):
                print(f"{i}. {prompt}")
            
            try:
                idx = int(input("\n번호를 선택하세요 (1-5): ")) - 1
                if 0 <= idx < len(sample_prompts):
                    selected_prompt = sample_prompts[idx]
                    print(f"\n질문: {selected_prompt}")
                    print("🤖 생성 중...")
                    
                    response = chat_bot.generate_text(selected_prompt)
                    print(f"\n🌸 Bllossom:\n{response}")
                else:
                    print("❌ 잘못된 번호입니다.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
                
        elif choice == "3":
            # 대화형 채팅
            print("\n💬 대화를 시작합니다. 'quit' 또는 '종료'를 입력하면 끝납니다.")
            print("-" * 50)
            
            conversation_count = 0
            while True:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    print("👋 대화를 종료합니다.")
                    break
                
                if not user_input:
                    print("💡 질문을 입력해주세요.")
                    continue
                
                print("🤖 생각 중...", end="", flush=True)
                
                try:
                    # 온도를 조금씩 변화시켜서 다양한 답변 생성
                    temp = 0.7 + (conversation_count % 3) * 0.1
                    response = chat_bot.generate_text(user_input, temperature=temp)
                    print(f"\r🌸 Bllossom: {response}")
                    
                    conversation_count += 1
                    
                except Exception as e:
                    print(f"\r❌ 생성 오류: {e}")
        
        else:
            print("❌ 잘못된 선택입니다.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ 프로그램을 중단합니다.")
    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {e}")

def quick_test():
    """빠른 테스트 함수 (오류 복구 포함)"""
    print("🧪 빠른 테스트 실행 중...")
    
    try:
        # 가장 가벼운 모델로 테스트
        chat_bot = SimpleBlossomChat("solar")  # SOLAR 모델 사용
        
        test_prompts = [
            "안녕하세요!",
            "파이썬이란 무엇인가요?",
            "좋은 하루 보내세요!"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔍 테스트 {i}: {prompt}")
            try:
                response = chat_bot.generate_text(prompt, max_length=100)
                print(f"🌸 응답: {response}")
            except Exception as e:
                print(f"❌ 테스트 {i} 실패: {e}")
                
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        print("💡 해결책: pip install transformers torch --upgrade")

def batch_generation():
    """여러 질문 배치 처리"""
    print("📦 배치 생성 모드")
    
    try:
        chat_bot = SimpleBlossomChat()
        
        questions = []
        print("질문들을 입력하세요 (빈 줄로 종료):")
        
        while True:
            question = input(f"질문 {len(questions)+1}: ").strip()
            if not question:
                break
            questions.append(question)
        
        if not questions:
            print("❌ 질문이 없습니다.")
            return
        
        print(f"\n🔄 {len(questions)}개 질문 처리 중...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 질문 {i}: {question}")
            response = chat_bot.generate_text(question)
            print(f"🌸 답변: {response}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ 배치 처리 실패: {e}")

if __name__ == "__main__":
    # 실행 모드 선택
    import sys
    
    print("🌸 한국어 AI 텍스트 생성기")
    print("=" * 30)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "test":
            quick_test()
        elif mode == "batch":
            batch_generation()
        else:
            main()
    else:
        # 안전한 시작을 위한 시스템 체크
        print("🔍 시스템 체크 중...")
        
        try:
            # PyTorch 체크
            print(f"✅ PyTorch: {torch.__version__}")
            print(f"✅ CUDA 사용 가능: {torch.cuda.is_available()}")
            
            # Transformers 체크
            import transformers
            print(f"✅ Transformers: {transformers.__version__}")
            
            print("\n🚀 모든 라이브러리가 준비되었습니다!")
            main()
            
        except ImportError as e:
            print(f"❌ 라이브러리 누락: {e}")
            print("\n💡 다음 명령어로 설치하세요:")
            print("pip install torch transformers accelerate")
        except Exception as e:
            print(f"❌ 시스템 체크 실패: {e}")
            print("🔄 기본 모드로 실행 시도...")
            main()