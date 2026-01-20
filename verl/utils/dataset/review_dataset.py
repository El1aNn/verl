import pandas as pd
import pyarrow.parquet as pq

def modify_parquet_ground_truth(input_file, output_file, target_index=124, new_ground_truth="12.7"):
    """
    读取parquet文件，修改指定行的ground_truth字段，并保存为新文件
    
    参数:
        input_file: 输入parquet文件路径
        output_file: 输出parquet文件路径
        target_index: 要修改的行的index值（默认124）
        new_ground_truth: 新的ground_truth值（默认"12.7"）
    """
    try:
        # 读取parquet文件
        df = pd.read_parquet(input_file)
        
        # 找到index等于target_index的行
        row_mask = df['extra_info'].apply(lambda x: x.get('index')) == target_index
        
        if not row_mask.any():
            print(f"警告：未找到index为{target_index}的行")
            return
        
        # 修改ground_truth字段
        df.loc[row_mask, 'reward_model'] = df.loc[row_mask, 'reward_model'].apply(
            lambda x: {**x, 'ground_truth': new_ground_truth}
        )
        
        # 保存修改后的文件
        df.to_parquet(output_file, index=False)
        print(f"修改完成！已将index={target_index}的行的ground_truth修改为{new_ground_truth}")
        print(f"修改后的文件已保存至：{output_file}")
        
        # 验证修改结果
        verify_df = pd.read_parquet(output_file)
        modified_row = verify_df[row_mask]['reward_model'].iloc[0]
        print(f"验证结果：修改后的ground_truth = {modified_row['ground_truth']}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"错误：处理过程中出现异常 - {str(e)}")

# ------------------- 配置参数 -------------------
# 请替换为你的实际文件路径
INPUT_PARQUET_FILE = "/root/rl/verl/data/dsr_sub/pi1.parquet"   # 输入文件路径
OUTPUT_PARQUET_FILE = "/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet"    # 输出文件路径
# ------------------------------------------------

# 执行修改
if __name__ == "__main__":
    modify_parquet_ground_truth(INPUT_PARQUET_FILE, OUTPUT_PARQUET_FILE)