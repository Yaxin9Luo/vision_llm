from codebook_mapping_analysis import load_imagenet_class_names

def main():
    # 加载类别名称
    class_names = load_imagenet_class_names()
    
    # 打印加载的类别数量
    print(f'加载了 {len(class_names)} 个类别')
    
    # 打印前5个类别
    print('前5个类别:')
    for i, (k, v) in enumerate(list(class_names.items())[:5]):
        print(f'{k}: {v}')

if __name__ == "__main__":
    main() 