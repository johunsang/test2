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
    def __init__(self):
        """간단한 한국어 텍스트 생성기 초기화"""
        self.model_name = "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🌸 Bllossom 모델 로딩 중... (디바이스: {self.device})")
        
        self.load_model()
    
    def load_model(self):
        """모델과 토크나이저 로드"""
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 (메모리 절약 설정)
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            # GPU 메모리 최적화 (GPU가 있는 경우)
            if torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                except ImportError:
                    print("⚠️ bitsandbytes가 설치되지 않음. 일반 로딩 사용")
                    model_kwargs["device_map"] = "auto"
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            print("✅ 모델 로딩 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("💡 인터넷 연결을 확인하거나 다른 모델을 시도해보세요.")
            raise
    
    def generate_text(self, prompt, max_length=300, temperature=0.7):
        """텍스트 생성 함수"""
        # Llama 3.1 채팅 포맷
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # 토크나이징
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # GPU로 이동 (필요한 경우)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # 디코딩 및 정리
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return response

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
    """빠른 테스트 함수"""
    print("🧪 빠른 테스트 실행 중...")
    
    try:
        chat_bot = SimpleBlossomChat()
        
        test_prompts = [
            "안녕하세요!",
            "파이썬이란 무엇인가요?",
            "좋은 하루 보내세요!"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔍 테스트 {i}: {prompt}")
            response = chat_bot.generate_text(prompt, max_length=100)
            print(f"🌸 응답: {response}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

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
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "test":
            quick_test()
        elif mode == "batch":
            batch_generation()
        else:
            main()
    else:
        main()