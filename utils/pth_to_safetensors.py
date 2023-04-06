import torch
from safetensors.torch import save_file

MODEL_NAME = './RWKV-4-Pile-430M-20220808-8066'

def main():
    input = f'{MODEL_NAME}.pth'
    output = f'{MODEL_NAME}.safetensors'
    print(f'* Loading with Torch: {input}')
    model = torch.load(input, map_location = 'cpu')
    print(f'* Saving with SafeTensors: {output}.safetensors')
    save_file(model, output)
    print('* Done.')

if __name__ == '__main__':
    main()
