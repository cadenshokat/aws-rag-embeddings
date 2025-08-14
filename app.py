import src.processing.prepare as prepare
import src.eval.run_base_eval as base_eval
import src.training.train as train
import src.eval.run_ft_eval as ft_eval

def main():
    print("=== Step 1: Preparing dataset ===")
    prepare.main()

    print("\n=== Step 2: Base model evaluation ===")
    base_eval.main()

    print("\n=== Step 3: Training model ===")
    train.main()

    print("\n=== Step 4: Fine-tuned model evaluation ===")
    ft_eval.main()

    print("\n✅ All steps completed.")

if __name__ == "__main__":
    main()
