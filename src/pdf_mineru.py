import requests # 导入requests库，用于发送HTTP请求
import time # 导入time库，用于控制程序等待时间
import zipfile # 导入zipfile库，用于解压zip文件

# MinerU平台的API密钥，用于身份验证（JWT Token格式）
api_key = 'eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NjQwMDUxNCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MjUyNzQ1MCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTc3MTcxOTIwNDciLCJvcGVuSWQiOm51bGwsInV1aWQiOiJiYjIzYWFkNS01NWM2LTQ5OTItYWFmMi1kZjE0ZmNkMmUyMDIiLCJlbWFpbCI6IiIsImV4cCI6MTc5MDMwMzQ1MH0.MAbu6a9OBP9vtY83kcDgCsKtnTY_OWs8GgIC151QudTFr3Bjm9yFP3NGvmpMaj5Uip6BE3TjU-5jdTZ-DsvlLg'

# 提交PDF解析任务，获取任务ID
def get_task_id(file_name):
    # MinerU平台的任务提交接口地址
    url='https://mineru.net/api/v4/extract/task'
    # 设置HTTP请求头，包含内容类型和身份验证信息
    header = {
        'Content-Type':'application/json', # 指定请求体为JSON格式
        "Authorization":f"Bearer {api_key}".format(api_key) # 使用Bearer Token进行身份认证
    }
    # 拼接PDF文件的完整URL地址（文件存储在阿里云OSS上）
    pdf_url = 'https://down.wss.show/onlobol/k/8t/k8tdonlobol?cdn_sign=1782531646-10-0-36dbebeb38fed7b69fbfdbc755e9ba12&exp=240&response-content-disposition=attachment%3B%20filename%3D%22%E3%80%90%E8%B4%A2%E6%8A%A5%E3%80%91%E4%B8%AD%E8%8A%AF%E5%9B%BD%E9%99%85%EF%BC%9A%E4%B8%AD%E8%8A%AF%E5%9B%BD%E9%99%852024%E5%B9%B4%E5%B9%B4%E5%BA%A6%E6%8A%A5%E5%91%8A-new.pdf%22%3B%20filename' + file_name
    # 构造请求体数据，包含解析任务的配置参数
    data = {
        'url':pdf_url, # PDF文件的URL地址
        'is_ocr':True, # 开启OCR识别功能，用于识别图片中的文字
        'enable_formula': False, # 关闭公式识别功能
    }

    # 发送POST请求提交解析任务
    res = requests.post(url,headers=header,json=data)
    # 打印HTTP响应状态码（200表示成功）
    print(res.status_code)
    # 打印完整的JSON响应内容
    print(res.json())
    # 打印响应中的data字段
    print(res.json()["data"])
    # 从响应中提取任务ID
    task_id = res.json()["data"]['task_id']
    # 返回任务ID，用于后续查询任务状态
    return task_id

# 根据任务ID轮询查询解析结果，任务完成后自动下载
def get_result(task_id):
    # 构造任务状态查询接口URL，拼接任务ID
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    # 设置HTTP请求头，与提交任务时相同
    header = {
        'Content-Type':'application/json', # 指定请求体为JSON格式
        "Authorization":f"Bearer {api_key}".format(api_key) # 使用Bearer Token进行身份认证
    }

    # 进入无限循环，持续轮询任务状态直到任务完成或出错
    while True:
        # 发送GET请求查询任务状态
        res = requests.get(url, headers=header)
        # 从响应中提取data字段
        result = res.json()["data"]
        # 打印当前任务状态信息
        print(result)
        # 获取任务的当前状态（pending/running/done等）
        state = result.get('state')
        # 获取任务的错误信息，如果没有则返回空字符串
        err_msg = result.get('err_msg', '')
        # 如果任务还在进行中，等待后重试
        if state in ['pending', 'running']:
            print("任务未完成，等待5秒后重试...")
            # 等待5秒，避免频繁请求
            time.sleep(5)
            # 跳过本次循环，继续下一次轮询
            continue
        # 如果有错误，输出错误信息
        if err_msg:
            # 打印错误信息
            print(f"任务出错: {err_msg}")
            # 出错时直接返回，结束函数
            return
        # 如果任务完成，下载文件
        if state == 'done':
            # 从结果中获取完整的zip文件下载URL
            full_zip_url = result.get('full_zip_url')
            # 如果下载URL存在
            if full_zip_url:
                # 生成本地保存的文件名，以任务ID命名
                local_filename = f"{task_id}.zip"
                # 打印下载开始提示
                print(f"开始下载: {full_zip_url}")
                # 以流式方式下载文件，适合大文件
                r = requests.get(full_zip_url, stream=True)
                # 以二进制写入模式打开本地文件
                with open(local_filename, 'wb') as f:
                    # 分块读取下载内容，每次读取8KB
                    for chunk in r.iter_content(chunk_size=8192):
                        # 确保chunk不为空
                        if chunk:
                            # 将数据块写入文件
                            f.write(chunk)
                # 打印下载完成提示
                print(f"下载完成，已保存到: {local_filename}")
                # 下载完成后自动解压
                unzip_file(local_filename)
            else:
                # 如果没有找到下载URL，打印提示信息
                print("未找到 full_zip_url，无法下载。")
            # 任务处理完成，退出函数
            return
        # 其他未知状态
        # 打印未知状态信息
        print(f"未知状态: {state}")
        # 未知状态时退出函数
        return

# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    # 如果没有指定解压目录，则使用zip文件名（去掉.zip后缀）作为目标目录
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    # 以读取模式打开zip文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 将zip文件中的所有内容解压到目标目录
        zip_ref.extractall(extract_dir)
    # 打印解压完成提示
    print(f"已解压到: {extract_dir}")

# 主程序入口，当直接运行此脚本时执行
if __name__ == "__main__":
    # 指定要处理的PDF文件名
    file_name = '【财报】中芯国际：中芯国际2024年年度报告-new.pdf'
    # 调用get_task_id函数提交任务，获取任务ID
    task_id = get_task_id(file_name)
    # 打印获取到的任务ID
    print('task_id:',task_id)
    # 调用get_result函数轮询任务状态并下载结果
    get_result(task_id)
