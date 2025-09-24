# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from typing import List, Union


from qwen_agent.agents import FnCallAgent

from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool


from src.prediction.predictor import predict_image

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


@register_tool('detect_img')
class DetectImag(BaseToolWithFileAccess):
  
    description = '图片检测工具，调用API实现图片检测'
    parameters = [
        {
            'name': 'filepath',
            'type': 'string',
            'description': '文件路径',
            'required': True
        },
        {
            'name': 'filename',
            'type': 'string',
            'description': '文件名称',
            'required': True
        }
        
    ]

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        super().call(params=params, files=files)
        params = self._verify_json_format_args(params)
        filename = str(params.get('filename', '')).strip()
        filepath = str(params.get('filepath', '')).strip()
        
        print(f"DetectImag function called with filename: {filename}, filepath: {filepath}")
        
        # 预测图像
        result = predict_image(filepath)
        
        # 添加文件路径到结果
        result['image_path'] = 'uploads/' + filename
        
        print(f"DetectImag function result: {result}")
        
        # 返回JSON字符串而不是字典
        import json
        return json.dumps(result, ensure_ascii=False)

def init_agent_service():
    llm_cfg_vl = {
        # Using Qwen2-VL deployed at any openai-compatible service such as vLLM:
        'model_type': 'qwenvl_oai',
        'model': 'qwen-vl-plus',
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # api_base
        'api_key': 'sk-a1db36ecb06c48888be66b773aa6bb68',
        'generate_cfg': {
            'max_retries': 10,
            #'fncall_prompt_type': 'qwen'  # 启用qwen function calling提示格式
        }

        # Using Qwen2-VL provided by Alibaba Cloud DashScope's openai-compatible service:
        # 'model_type': 'qwenvl_oai',
        # 'model': 'qwen-vl-max-0809',
        # 'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

        # Using Qwen2-VL provided by Alibaba Cloud DashScope:
        # 'model_type': 'qwenvl_dashscope',
        # 'model': 'qwen-vl-max-0809',
        # 'api_key': os.getenv('DASHSCOPE_API_KEY'),
        # 'generate_cfg': {
        #     'max_retries': 10,
        #     'fncall_prompt_type': 'qwen'
        # }
    }

    tools = [
        'detect_img'
    ]  # code_interpreter is a built-in tool in Qwen-Agent

    system_instruction = """你是一个图像检测Agent，你专门根据用户提供的图片路径进行图像真伪检测。
请按照以下步骤工作：
1. 调用detect_img函数检测图片
2. 基于检测结果生成详细的分析报告
3. 确保只调用一次detect_img函数，避免重复调用"""
    
    bot = FnCallAgent(
        llm=llm_cfg_vl,
        name='Qwen2-VL',
        system_message=system_instruction,
        description='function calling',
        function_list=tools,
    )

    return bot


def test():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = [{
        'role':
            'user',
        'content': [
            {
                'image': os.path.abspath(os.path.join(ROOT_RESOURCE, 'screenshot_with_plot.jpeg'))
            },
            {
                'text': '调用工具放大右边的表格'
            },
        ],
    }]

    for response in bot.run(messages=messages):
        print('bot response:', response)

