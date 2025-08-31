import os
import re
import json
import warnings  # Add this missing import
import torch
import openai
import argparse
import tiktoken
import logging
import transformers
import time
from google import genai
from tqdm import tqdm

# Suppress noisy NumPy-related warnings triggered indirectly by torch import
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Failed to initialize NumPy: _ARRAY_API not found"
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"numpy(\.|$)")


# Clear any proxy environment variables that might cause issues
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy',
              'https_proxy', 'ALL_PROXY', 'all_proxy']
for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]

# Mitigate CUDA memory fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Load API keys from environment variables or .env file
# Make sure to set these in your environment or create a .env file
# export OPENAI_API_KEY="your-api-key-here"
# export OPEN_API_URL="https://api.openai.com/v1"
# export GOOGLE_API_KEY="your-google-api-key-here"

# Check if API key is set in environment
if not os.environ.get('OPENAI_API_KEY') and not os.environ.get('GOOGLE_API_KEY'):
    raise ValueError("Either OPENAI_API_KEY or GOOGLE_API_KEY environment variable must be set. Please set one before running this script.")

# Set default API URL if not provided
if not os.environ.get('OPEN_API_URL'):
    os.environ["OPEN_API_URL"] = "https://api.openai.com/v1"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

model_list = [
    'llama-3-8b', 
    'llama-3-70b', 
    'gpt-3.5-turbo-0125', 
    'gpt-3.5-turbo', 
    'gpt-4-turbo', 
    'gpt-4-turbo-2024-04-09',
    'gpt-4o', 
    'gpt-4o-2024-05-13',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gemini-1.0-pro',
]

OPEN_API_KEY = os.environ.get('OPENAI_API_KEY')
OPEN_API_URL = os.environ.get('OPEN_API_URL')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE_ID = 0

client = None
pipeline = None
genai_client = None


def num_tokens_from_message(messages, model='gpt-3.5-turbo'):
    ''' Return the number of tokens used by a list of messages. '''

    # Handle Gemini models differently since tiktoken doesn't support them
    if 'gemini' in model:
        # For Gemini, use a simple approximation based on words
        total_text = ""
        for message in messages:
            for key, value in message.items():
                if key == 'content':
                    total_text += value + " "
        # Rough approximation: 1 token ≈ 0.75 words
        return int(len(total_text.split()) * 1.33)
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Try to use a compatible encoding
        try:
            encoding = tiktoken.get_encoding('cl100k_base')  # GPT-4 encoding
        except ValueError:
            try:
                encoding = tiktoken.get_encoding('p50k_base')  # GPT-3 encoding
            except ValueError:
                # Fallback to simple word-based approximation
                logging.warning(f'No compatible encoding found for {model}. Using word-based approximation.')
                total_text = ""
                for message in messages:
                    for key, value in message.items():
                        if key == 'content':
                            total_text += value + " "
                return int(len(total_text.split()) * 1.33)
    
    tokens_per_message = 3
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == 'role':
                num_tokens += tokens_per_message
    num_tokens += 3 # every reply is primed with <|start|>assistant<|message|>

    return num_tokens


def num_tokens_from_text(text, model='gpt-3.5-turbo'):
    ''' Return the number of tokens used by a text string. '''
    
    # Handle Gemini models differently
    if 'gemini' in model:
        # Simple approximation: 1 token ≈ 0.75 words
        return int(len(text.split()) * 1.33)
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Try to use a compatible encoding
        try:
            encoding = tiktoken.get_encoding('cl100k_base')  # GPT-4 encoding
        except ValueError:
            try:
                encoding = tiktoken.get_encoding('p50k_base')  # GPT-3 encoding
            except ValueError:
                # Fallback to simple word-based approximation
                logging.warning(f'No compatible encoding found for {model}. Using word-based approximation.')
                return int(len(text.split()) * 1.33)
    
    return len(encoding.encode(text))


def get_prompt(item: dict, prompt_dict: dict, psg_num: int=5, doc_key: str='text'):
    ''' generate prompt '''

    if all(key in prompt_dict for key in ['instruction', 'demo_sep', 'demo_prompt', 'doc_prompt']):

        if all(key in item for key in ['question', 'docs']):
            prompt = prompt_dict['demo_prompt'].replace('{INST}', prompt_dict['instruction'])
            prompt = prompt.replace('{Q}', item['question'])

            doc_prompt_list = []
            for doc_id, doc in enumerate(item['docs'][:psg_num]):
                doc_key = doc_key if doc_key in doc else 'text'

                doc_prompt = prompt_dict['doc_prompt'].replace('{ID}', str(doc_id + 1))
                doc_prompt = doc_prompt.replace('{T}', doc['title'])
                doc_prompt = doc_prompt.replace('{P}', doc[doc_key])
                doc_prompt_list.append(doc_prompt)
            
            prompt = prompt.replace('{D}', prompt_dict['demo_sep'].join(doc_prompt_list))
            prompt = prompt.replace('{A}', '').rstrip()

            return prompt
        else:
            logging.error('The data item has some mistakes.')
            return None
    else:
        logging.error('The prompt config has some mistakes.')
        return None


def get_messages(
        prompt: str, 
        instruction: str='You are ChatGPT, a large language model trained by OpenAI.'
    ) -> list:
    ''' generate messages '''
    
    messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': prompt}
    ]

    return messages


def generate_openai(messages: list, model_name: str, temperature: float=0.5) -> str:
    ''' using api to get openai model response '''

    global client

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )

        return completion.choices[0].message.content, completion.usage.prompt_tokens # number of input tokens
    
    except Exception as e:
        logging.error(f'Exception occurred during calling {model_name}')
        return '', 0


def generate_llama_3(messages: list, temperature: float=0.5) -> str:
    ''' generate response using llama-3 '''

    global pipeline

    # Check if chat template exists, if not create a simple prompt
    try:
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except ValueError:
        # Fallback: manually create prompt if no chat template
        logging.warning("No chat template found, using manual prompt formatting")
        prompt = ""
        for message in messages:
            if message['role'] == 'system':
                prompt += f"System: {message['content']}\n\n"
            elif message['role'] == 'user':
                prompt += f"User: {message['content']}\n\nAssistant: "
        
    terminators = []
    if hasattr(pipeline.tokenizer, 'eos_token_id') and pipeline.tokenizer.eos_token_id:
        terminators.append(pipeline.tokenizer.eos_token_id)
    
    # Try to add Llama-specific terminators if they exist
    try:
        eot_id = pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != pipeline.tokenizer.unk_token_id:  # Only add if token exists
            terminators.append(eot_id)
    except:
        pass  # Token doesn't exist, continue without it

    result = pipeline(
        prompt,
        do_sample=True,
        eos_token_id=terminators if terminators else None,
        remove_invalid_values=True,
        top_k=10,
        num_return_sequences=1,
        max_new_tokens=400,
        temperature=temperature,
        pad_token_id=pipeline.tokenizer.eos_token_id,  # Set pad token
    )

    # Extract response text
    generated_text = result[0]['generated_text']
    
    # Try to extract assistant response using different patterns
    patterns = [
        r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*)',
        r'Assistant: (.*)',
        r'Response: (.*)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, generated_text.replace('\n', ' '), re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    # If no pattern matches, return text after the original prompt
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
        return response
    
    return generated_text.strip()


def generate_gemini(prompt: str, model_name: str, temperature: float=0.5) -> str:
    ''' generate response using Gemini API using new Google GenAI SDK with rate limiting '''
    
    global genai_client
    
    max_retries = 3
    base_delay = 60  # Start with 60 seconds delay for rate limits
    
    for attempt in range(max_retries):
        try:
            if genai_client is None:
                genai_client = genai.Client()
            
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=400,
                    top_k=10,
                    top_p=0.9,
                )
            )
            
            return response.text, len(prompt.split())  # Approximate token count
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if '429' in error_str and 'RESOURCE_EXHAUSTED' in error_str:
                if attempt < max_retries - 1:
                    # Extract retry delay from error if available
                    if 'retryDelay' in error_str:
                        try:
                            # Parse retry delay (usually in seconds)
                            import re
                            delay_match = re.search(r'"retryDelay":\s*"(\d+)s"', error_str)
                            if delay_match:
                                delay = int(delay_match.group(1)) + 10  # Add buffer
                            else:
                                delay = base_delay
                        except:
                            delay = base_delay
                    else:
                        delay = base_delay
                    
                    logging.warning(f'Rate limit hit for {model_name}, waiting {delay} seconds before retry {attempt + 1}/{max_retries}')
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f'Rate limit exceeded for {model_name} after {max_retries} retries')
                    return '', 0
            else:
                logging.error(f'Exception occurred during calling {model_name}: {error_str}')
                return '', 0
    
    return '', 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='The dataset file.')
    parser.add_argument('--prompt_config', required=True, help='The prompt configuration file.')
    parser.add_argument('--output_path', required=True, help='The file of generated responses.')
    parser.add_argument('--model_name', required=True, choices=model_list, help='The name of model.')
    parser.add_argument('--no_call', action='store_true', help='Just calculate predicted number of total tokens. No calling api.')
    parser.add_argument('--psg_num', type=int, default=5, help='The number of passages used in generation.')
    parser.add_argument('--use_sum', action='store_true', help='Use passage\'s summary.')
    parser.add_argument('--use_snippet', action='store_true', help='Use passages\' snippet.')
    args = parser.parse_args()

    with open(args.data_path, 'r') as file:
        data = json.load(file)

    with open(args.prompt_config, 'r') as file:
        prompt_dict = json.load(file)

    total_tokens = 0
    
    total_tokens = 0
    for item in tqdm(data):
        if args.no_call:
            if 'gemini' in args.model_name:
                prompt = get_prompt(item, prompt_dict, args.psg_num)
                if prompt:
                    instruction = 'You are a helpful AI assistant for QA task.'
                    full_prompt = f"{instruction}\n\n{prompt}"
                    total_tokens += num_tokens_from_text(full_prompt, args.model_name)
            else:
                total_tokens += num_tokens_from_message(get_messages(get_prompt(item, prompt_dict)), args.model_name)
        
        else:
            if 'gpt' in args.model_name:
                # using openai
                global client

                if client is None:
                    client = openai.OpenAI(
                        base_url=OPEN_API_URL,
                        api_key=OPEN_API_KEY
                    )

                instruction = 'You are ChatGPT, a large language model trained by OpenAI.'

                if args.use_sum:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'summary')
                elif args.use_snippet:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'extraction')
                else:
                    prompt = get_prompt(item, prompt_dict, args.psg_num)
                
                messages = get_messages(prompt, instruction)

                output, num_input_tokens = generate_openai(messages, args.model_name)
            
            elif 'gemini' in args.model_name:
                # using gemini
                instruction = 'You are a helpful AI assistant for QA task.'

                if args.use_sum:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'summary')
                elif args.use_snippet:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'extraction')
                else:
                    prompt = get_prompt(item, prompt_dict, args.psg_num)
                
                # For Gemini, we'll use the prompt directly instead of messages
                full_prompt = f"{instruction}\n\n{prompt}"
                output, num_input_tokens = generate_gemini(full_prompt, args.model_name)
                # Update token count for Gemini
                if num_input_tokens == 0:
                    num_input_tokens = num_tokens_from_text(full_prompt, args.model_name)
            
            else:
                # using llama
                global pipeline

                if pipeline is None:
                    pipeline = transformers.pipeline(
                        'text-generation',
                        model=args.model_name,
                        torch_dtype=torch.float16,
                        device_map='auto'
                    )

                instruction = 'You are a helpful AI assistant for QA task.'

                if args.use_sum:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'summary')
                elif args.use_snippet:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'extraction')
                else:
                    prompt = get_prompt(item, prompt_dict, args.psg_num)
                
                messages = get_messages(prompt, instruction)

                num_input_tokens = num_tokens_from_message(messages)
                output = generate_llama_3(messages)

            item['output'] = output
            total_tokens += num_input_tokens
    
    if args.no_call:
        logging.info(f'The predicted number of tokens is {total_tokens}')
    else:
        logging.info(f'The actual number of tokens is {total_tokens}')

        with open(args.output_path, 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()