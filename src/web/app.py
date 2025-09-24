import logging
import os
import sys
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, Response
from werkzeug.utils import secure_filename
from qwen_agent.utils.output_beautify import typewriter_print

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.ai.qwen2vl_assistant_tooluse import init_agent_service

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.predictor import load_model, predict_image

# 创建Flask应用
app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
app.secret_key = 'image_forensics_detector_key'

# 配置上传文件目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上传大小

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        flash('未选择文件')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('未选择文件')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 安全地获取文件名并保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预测图像
        result = predict_image(filepath)
        
        # 添加文件路径到结果
        result['image_path'] = 'uploads/' + filename
        
        return render_template('result.html', result=result)
    
    flash('不支持的文件类型，请上传JPG、JPEG或PNG格式的图像')
    return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API端点用于预测"""
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 安全地获取文件名并保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预测图像
        result = predict_image(filepath)
        
        # 添加文件路径到结果
        result['image_path'] = 'uploads/' + filename
        
        return jsonify(result)
    
    return jsonify({'error': '不支持的文件类型，请上传JPG、JPEG或PNG格式的图像'}), 400

@app.route('/api/ai/predict/stream', methods=['POST'])
def api_ai_predict_stream():
    """API端点用于AI预测流式响应"""
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    
    file = request.files['file']
    query = request.form.get('query', '请分析这张图片是否为伪造图片，并详细说明理由')
    
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 安全地获取文件名并保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        def generate():
            try:
                # 状态管理变量
                phase = "init"  # init -> function_call -> function_result -> assistant_content
                last_content = ""
                
                # 初始化服务
                yield "data: 🚀 步骤1：正在初始化AI服务...\n\n"
                
                bot = init_agent_service()
                yield "data: ✅ 步骤1完成：AI服务初始化成功\n\n"
                
                # 准备消息
                messages = [{
                    'role': 'user',
                    'content': [
                        {'image': os.path.abspath(filepath)},
                        {'text': f'{query},filename:{filename},filepath:{filepath}'}
                    ],
                }]
                
                yield "data: 🤖 步骤2：正在启动AI Agent分析...\n\n"
                
                for response in bot.run(messages=messages):
                    if isinstance(response, list) and len(response) > 0:
                        latest = response[-1]
                        
                        # 阶段1：检测到function_call
                        if phase == "init" and isinstance(latest, dict) and 'function_call' in latest:
                            phase = "function_call"
                            yield "data: ✅ 步骤2完成：AI Agent已启动\n\n"
                            yield "data: 📁 步骤3：正在分析图像参数...\n\n"
                        
                        # 阶段2：function_call arguments完整
                        elif phase == "function_call" and isinstance(latest, dict):
                            if 'function_call' in latest:
                                args = latest.get('function_call', {}).get('arguments', '')
                                if args and args.strip():
                                    try:
                                        import json
                                        args_dict = json.loads(args)
                                        if args_dict.get('filepath') and args_dict.get('filename'):
                                            yield "data: ✅ 步骤3完成：参数分析完成\n\n"
                                            yield "data: 🔍 步骤4：正在执行图像检测...\n\n"
                                            phase = "function_result"
                                    except json.JSONDecodeError:
                                        pass
                            elif latest.get('role') == 'function':
                                # 检测结果返回
                                yield "data: ✅ 步骤4完成：图像检测完成\n\n"
                                yield "data: 💭 步骤5：正在生成AI分析报告...\n\n"
                                phase = "assistant_content"
                        
                        # 阶段3：assistant content 流式输出
                        elif phase == "assistant_content" and isinstance(latest, dict) and latest.get('role') == 'assistant':
                            current_content = latest.get('content', '')
                            if current_content:
                                # 计算增量内容
                                if current_content != last_content:
                                    if last_content and current_content.startswith(last_content):
                                        # 发送增量部分
                                        new_content = current_content[len(last_content):]
                                        if new_content:
                                            yield f"data: INCREMENT:{new_content}\n\n"
                                    else:
                                        # 发送完整内容（第一次或完全不同）
                                        yield f"data: FULL:{current_content}\n\n"
                                    
                                    last_content = current_content
                        
                        # 检测assistant role开始（从function_result阶段切换）
                        elif phase == "function_result" and isinstance(latest, dict) and latest.get('role') == 'assistant':
                            current_content = latest.get('content', '')
                            if current_content:
                                yield "data: ✅ 步骤4完成：图像检测完成\n\n"
                                yield "data: 💭 步骤5：正在生成AI分析报告...\n\n"
                                phase = "assistant_content"
                                # 发送第一部分content
                                yield f"data: FULL:{current_content}\n\n"
                                last_content = current_content
                
                yield "data: ✅ 步骤5完成：AI分析报告生成完成\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logging.error(f"Error in generate: {e}", exc_info=True)
                yield f"data: ❌ 错误：{str(e)}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype='text/plain', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        })
    
    return jsonify({'error': '不支持的文件类型，请上传JPG、JPEG或PNG格式的图像'}), 400


def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='图像伪造检测Web应用')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--model-name', type=str, default='efficientnet_b0', help='模型名称')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model_path, args.model_name)
    
    # 启动应用
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main() 