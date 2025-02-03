import argparse

import torch
from lorem_text import lorem

from model.trans_model import Transformer
from model.utils.utils import batch
from tokenizers.bpe_tokenizer import Tokenizer
from tokenizers.char_tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Parameters to run my vSLM.")
    parser.add_argument(
        "--training_iterations",
        type=int,
        default="5000",
        help="Number of training iterations",
    )
    parser.add_argument(
        "--text",
        type=str,
        choices=["lorem", "shakespeare"],
        default="shakespeare",
        help="Lorem ipsum or shakespeare",
    )
    parser.add_argument(
        "--train_model",
        default=True,
        help="Train the model if True, else load the model",
    )
    parser.add_argument(
        "--task",
        default="generation",
        type=str,        
        choices=["generation", "translation"],
        help="Task to perform: generation or translation (fr-en)",
    )

    args = parser.parse_args()
    mapping = {"True": True, "False": False}
    args.train_model = mapping[str(args.train_model)]
    # Train on mps
    device = torch.device("mps")

    if args.task == "generation":
        if args.text == "lorem":
            text = lorem.paragraphs(10000)
        elif args.text == "shakespeare":
            text = open("data/shakespeare.txt").read()
        else:
            exit()
    elif args.task == "translation":
        text_en = open("data/en2fr.txt").read()
        text_fr = open("data/fr2en.txt").read()
        # We manually set text to the distinct characters of the two languages. Will be used only to build the tokenizer.
        text = ''.join(set(text_en) | set(text_fr))
    # Initialize the tokenizer
    if False:
        # Load a trained tokenizer
        tok = Tokenizer()
        path = "./bpe/tokenizer/models"
        tok.load(path + ".model")
    else:
        # Use this one for simple training
        tok = CharTokenizer(text)

    # Parameters
    batch_size = 32
    block_size = 16
    n_head = 6
    n_embd = 32
    n_layer = 6
    vocab_size = tok.vocab_size
    training_iterations = args.training_iterations
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+

    model = Transformer(
        vocab_size=vocab_size,
        n_embed=n_embd,
        n_heads=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        training_iterations=training_iterations,
    ).to(device)

    print(model)
    if args.train_model:
        if args.task == "generation":
            # We split our text into 90% train and 10% validation
            train_data = text[: int(0.9 * len(text))]
            val_data = text[int(0.9 * len(text)) :]
            # Train the data
            model.fit_generation(
                training_data=tok.encode(train_data),
                block_size=block_size,
                batch_size=batch_size,
                eval_data=tok.encode(val_data),
            )
            model.save("weights/model_generation.pth")
        elif args.task == "translation":
            # We split our text into 90% train and 10% validation
            size = int(0.9*len(text_en.split("\n")))
            train_data = (
                    [tok.encode(elt) for elt in text_en.split("\n")[: size] ],
                    [tok.encode(elt) for elt in text_fr.split("\n")[: size] ],                    
            )

            val_data = (
                    [tok.encode(elt) for elt in text_en.split("\n")[size: ] ],
                    [tok.encode(elt) for elt in text_fr.split("\n")[size: ] ],                    
            )            
            
            model.fit_translation(
                training_data=train_data,                
                block_size=block_size,
                batch_size=batch_size,
                eval_data=val_data,
            )
            model.save("weights/model_translation.pth")
    else:
        if args.task == "generation":
            print("Generation model not trained, loading weights")
            model.load("weights/model_generation.pth")
        elif args.task == "translation":
            print("Translation model not trained, loading weights")
            model.load("weights/model_translation.pth")

    print("##### Example after Training #####")
    if args.task == "generation":
        x = torch.zeros((1, 1), dtype=torch.long).to(device)
        res = model.generate(x, 500)
        print("Decoded output message: ", tok.decode(res[0].detach().cpu().tolist()))
    print("##### End of generation example #####")


if __name__ == "__main__":
    main()
