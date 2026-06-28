import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed
import dashscope

# OpenAI基础处理器，封装了消息发送、结构化输出、计费等逻辑
class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'Qwen3.6-plus'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        # 加载OpenAI API密钥，初始化LLM
        load_dotenv(override=True)
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        # 发送消息到OpenAI，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        # 部分模型不支持temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            try:
                params["response_format"] = response_format
                completion = self.llm.beta.chat.completions.parse(**params)

                response = completion.choices[0].message.parsed
                if response is None:
                    content = None
                else:
                    content = response.dict()
            except Exception as e:
                print(f"[send_message] structured parse error: {e}")
                import traceback
                traceback.print_exc()
                content = None

        if 'completion' in locals() and completion is not None:
            self.response_data = {
                "model": completion.model if completion.model else "",
                "input_tokens": completion.usage.prompt_tokens if completion.usage and completion.usage.prompt_tokens else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage and completion.usage.completion_tokens else 0
            }
            print(self.response_data)
        else:
            self.response_data = {}

        return content

    def send_message_streaming(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
    ):
        # 发送消息到OpenAI，支持流式输出
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ],
            "stream": True
        }
        
        if "o3-mini" not in model:
            params["temperature"] = temperature

        # 当使用结构化输出时，OpenAI不允许在stream=True模式下使用response_format参数
        # 所以先不传response_format，流式收集文本后再手动解析
        # 注意：is_structured 参数仅用于决定是否进行后续的结构化解析，不影响流式发送
        params.pop("response_format", None)

        completion = self.llm.chat.completions.create(**params)
        
        accumulated_text = ""
        token_count = 0
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated_text += delta
                token_count += 1
                yield {"type": "token", "content": delta, "accumulated": accumulated_text}
        
        self.response_data = {
            "model": getattr(chunk, 'model', ''),
            "input_tokens": 0,
            "output_tokens": token_count
        }

        final_content = accumulated_text
        # 如果需要使用结构化输出，则在收集完所有token后进行解析
        if is_structured and response_format is not None and accumulated_text:
            try:
                repaired_json = repair_json(accumulated_text)
                parsed_dict = json.loads(repaired_json)
                validated_data = response_format.model_validate(parsed_dict)
                final_content = validated_data.model_dump()
            except Exception as e:
                print(f"[send_message_streaming] structured parse error from stream: {e}")
                try:
                    repaired_json = repair_json(accumulated_text)
                    final_content = json.loads(repaired_json)
                except Exception:
                    final_content = accumulated_text

        yield {"type": "done", "content": final_content}

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        # 统计字符串的token数
        encoding = tiktoken.get_encoding(encoding_name)
        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count


# IBM API基础处理器，支持余额查询、模型列表、嵌入、消息发送等
class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv(override=True)
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """查询当前API余额"""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """获取可用基础模型列表"""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """获取文本的向量嵌入"""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        # 发送消息到IBM API，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv(override=True)
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "dashscope"] ="dashscope"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        elif self.provider == "dashscope":
            self.processor = BaseDashscopeProcessor()

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def send_message_streaming(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message_streaming call to the appropriate processor.
        Yields incremental token updates and final result.
        """
        if model is None:
            model = self.processor.default_model
        yield from self.processor.send_message_streaming(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        print(f"[DEBUG] get_answer_from_rag_context: rag_context type={type(rag_context)}, model={model}, schema={schema}")
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        print(f"[DEBUG] send_message returned: answer_dict type={type(answer_dict)}, value={answer_dict}")
        self.response_data = self.processor.response_data
        
        # 检查返回结果是否为 None
        if answer_dict is None:
            return {
                "step_by_step_analysis": "",
                "reasoning_summary": "",
                "relevant_pages": [],
                "final_answer": "N/A",
                "error": "LLM returned None response"
            }
        
        # 检查返回的字典是否包含所需的字段，如果不是dashscope则进行兜底
        if not isinstance(answer_dict, dict) or 'step_by_step_analysis' not in answer_dict:
            print(f"[DEBUG] answer_dict is not expected format: {type(answer_dict)}")
            # 如果是dashscope返回的基本格式，尝试保留其内容
            if isinstance(answer_dict, dict) and 'final_answer' in answer_dict:
                # 这是dashscope处理后的格式，尝试从final_answer中提取结构化信息
                final_answer_content = answer_dict.get("final_answer", "N/A")
                
                # 如果final_answer是字符串且包含结构化信息，尝试解析
                if isinstance(final_answer_content, str) and final_answer_content.strip().startswith('{'):
                    try:
                        structured_data = json.loads(final_answer_content)
                        answer_dict = structured_data
                    except json.JSONDecodeError:
                        # 如果final_answer不是JSON，保持原有结构
                        answer_dict = {
                            "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                            "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                            "relevant_pages": answer_dict.get("relevant_pages", []),
                            "final_answer": answer_dict.get("final_answer", "N/A")
                        }
                else:
                    # 否则使用兜底结构
                    answer_dict = {
                        "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                        "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                        "relevant_pages": answer_dict.get("relevant_pages", []),
                        "final_answer": answer_dict.get("final_answer", "N/A")
                    }
            else:
                # 如果不是预期格式，进行兜底
                answer_dict = {
                    "step_by_step_analysis": "",
                    "reasoning_summary": "",
                    "relevant_pages": [],
                    "final_answer": "N/A"
                }
        print(f"[DEBUG] get_answer_from_rag_context final result: {answer_dict}")
        return answer_dict

    def get_answer_streaming(self, question, rag_context, schema, model):
        """
        流式获取答案，yield 以下阶段:
        - {"stage": "retrieval_done"} 
        - {"stage": "streaming", "token": delta, "accumulated": text}
        - {"stage": "parsing", "content": raw_text}
        - {"stage": "done", "answer_dict": full_structured_dict}
        - {"stage": "error", "error": error_msg}
        """
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        try:
            yield {"stage": "retrieval_done"}
            
            full_response = self.processor.send_message_streaming(
                model=model,
                system_content=system_prompt,
                human_content=user_prompt.format(context=rag_context, question=question),
                is_structured=True,
                response_format=response_format
            )
            
            accumulated = ""
            for chunk in full_response:
                if chunk.get("type") == "token":
                    yield {"stage": "streaming", "token": chunk["content"], "accumulated": chunk["accumulated"]}
                elif chunk.get("type") == "done":
                    final_content = chunk["content"]
                    self.response_data = self.processor.response_data
                    
                    # 解析最终的 JSON
                    answer_dict = None
                    if isinstance(final_content, dict):
                        answer_dict = final_content
                    elif isinstance(final_content, str):
                        try:
                            # 尝试解析 JSON
                            content_str = final_content.strip()
                            if content_str.startswith('{'):
                                answer_dict = json.loads(content_str)
                            elif content_str.startswith('```'):
                                first_bt = content_str.find('```') + 3
                                next_nl = content_str.find('\n', first_bt)
                                if next_nl > 0:
                                    first_bt = next_nl + 1
                                last_bt = content_str.rfind('```')
                                if last_bt > first_bt:
                                    json_str = content_str[first_bt:last_bt].strip()
                                else:
                                    json_str = content_str
                                answer_dict = json.loads(json_str)
                            else:
                                answer_dict = {"final_answer": content_str}
                        except (json.JSONDecodeError, TypeError):
                            answer_dict = {"final_answer": final_content}
                    
                    if answer_dict is None or not isinstance(answer_dict, dict) or 'step_by_step_analysis' not in answer_dict:
                        if isinstance(answer_dict, dict) and 'final_answer' in answer_dict:
                            final_ans_content = answer_dict.get("final_answer", "N/A")
                            if isinstance(final_ans_content, str) and final_ans_content.strip().startswith('{'):
                                try:
                                    answer_dict = json.loads(final_ans_content)
                                except json.JSONDecodeError:
                                    answer_dict = {
                                        "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                                        "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                                        "relevant_pages": answer_dict.get("relevant_pages", []),
                                        "final_answer": answer_dict.get("final_answer", "N/A")
                                    }
                            else:
                                answer_dict = {
                                    "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                                    "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                                    "relevant_pages": answer_dict.get("relevant_pages", []),
                                    "final_answer": answer_dict.get("final_answer", "N/A")
                                }
                        else:
                            answer_dict = {
                                "step_by_step_analysis": "",
                                "reasoning_summary": "",
                                "relevant_pages": [],
                                "final_answer": "N/A"
                            }
                    
                    yield {"stage": "done", "answer_dict": answer_dict}
                    return
                
        except Exception as e:
            yield {"stage": "error", "error": str(e)}


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        elif schema == "string":
            # 新增：支持开放性文本问题
            system_prompt = (prompts.AnswerWithRAGContextStringPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextStringPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextStringPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextStringPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="Qwen3.6-plus",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list

# DashScope基础处理器，支持Qwen大模型对话
class BaseDashscopeProcessor:
    def __init__(self):
        # 从环境变量读取API-KEY
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.default_model = 'Qwen3.6-plus'

    def send_message(
        self,
        model="deepseek-v4-pro",
        temperature=0.1,
        seed=None,  # 兼容参数，暂不使用
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        发送消息到DashScope Qwen大模型，支持 system_content + human_content 拼接为 messages。
        暂不支持结构化输出。
        """
        if model is None:
            model = self.default_model
        # 拼接 messages
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if human_content:
            messages.append({"role": "user", "content": human_content})
        # 调用 dashscope Generation.call
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format='message'
        )
        
        # 检查 response 是否为 None
        if response is None:
            error_msg = "DashScope API returned None response"
            self.response_data = {"model": model, "input_tokens": None, "output_tokens": None}
            return {"final_answer": "N/A", "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": [], "error": error_msg}
        
        # 检查是否有错误码
        if hasattr(response, 'status_code') and response.status_code != 200:
            error_msg = f"DashScope API error: status={response.status_code}, code={getattr(response, 'code', 'N/A')}, message={getattr(response, 'message', 'N/A')}"
            self.response_data = {"model": model, "input_tokens": None, "output_tokens": None}
            return {"final_answer": "N/A", "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": [], "error": error_msg}
        
        # 兼容 openai/gemini 返回格式，始终返回 dict
        if hasattr(response, 'output') and hasattr(response.output, 'choices') and response.output.choices:
            content = response.output.choices[0].message.content
        else:
            content = None
        # 增加 response_data 属性，保证接口一致性
        try:
            self.response_data = {"model": model, "input_tokens": response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else None, "output_tokens": response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else None}
        except Exception:
            self.response_data = {"model": model, "input_tokens": None, "output_tokens": None}
        
        # 如果 content 为 None，返回兜底结果
        if content is None:
            error_msg = "DashScope API returned empty content"
            return {"final_answer": "N/A", "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": [], "error": error_msg}
        
        # 尝试解析 content 为 JSON，如果是结构化响应
        try:
            content_str = content.strip()
            if content_str.startswith('```') and '```' in content_str[3:]:
                first_backtick = content_str.find('```') + 3
                next_newline = content_str.find('\n', first_backtick)
                if next_newline > 0:
                    first_backtick = next_newline + 1
                last_backtick = content_str.rfind('```')
                if last_backtick > first_backtick:
                    json_str = content_str[first_backtick:last_backtick].strip()
                else:
                    json_str = content_str
            else:
                json_str = content_str
            
            parsed_content = json.loads(json_str)
            return parsed_content
        except (json.JSONDecodeError, TypeError):
            print(f"Content is not valid JSON, returning basic format: {content}")
            return {"final_answer": content, "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": []}

    def send_message_streaming(
        self,
        model=None,
        temperature=0.1,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        发送消息到DashScope Qwen大模型，支持流式输出。
        使用 stream=True 参数获取增量响应。
        """
        if model is None:
            model = self.default_model
        
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if human_content:
            messages.append({"role": "user", "content": human_content})
        
        # 使用 stream=True 获取流式响应
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format='message',
            stream=True
        )
        
        accumulated_text = ""
        token_count = 0
        
        for chunk in response:
            # DashScope 流式响应的 output choices[0].message.content 包含增量文本
            if hasattr(chunk, 'output') and hasattr(chunk.output, 'choices') and chunk.output.choices:
                delta = chunk.output.choices[0].message.content
                if delta:
                    accumulated_text += delta
                    token_count += 1
                    yield {"type": "token", "content": delta, "accumulated": accumulated_text}
        
        self.response_data = {
            "model": model,
            "input_tokens": getattr(getattr(response, 'usage', None), 'input_tokens', None),
            "output_tokens": token_count
        }

        final_content = accumulated_text
        if is_structured and response_format is not None and accumulated_text:
            try:
                repaired_json = repair_json(accumulated_text)
                parsed_dict = json.loads(repaired_json)
                validated_data = response_format.model_validate(parsed_dict)
                final_content = validated_data.model_dump()
            except Exception as e:
                print(f"[send_message_streaming-dashscope] structured parse error from stream: {e}")
                try:
                    repaired_json = repair_json(accumulated_text)
                    final_content = json.loads(repaired_json)
                except Exception:
                    final_content = accumulated_text

        yield {"type": "done", "content": final_content}
