import os
import sys
import matplotlib.pyplot as plt

def get_bpseq_lengths(folder_path):
    lengths = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".bpseq"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r") as f:
                # Count non-empty, non-comment lines
                length = sum(
                    1 for line in f
                    if line.strip() and not line.strip().startswith("#")
                )
                lengths.append(length)
        except (OSError, IOError):
            print(f"Warning: could not read {filename}")

    return lengths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bpseq_length_histogram.py <bpseq_folder>")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print("Error: Provided path is not a directory.")
        sys.exit(1)

    lengths = get_bpseq_lengths(folder)

    if not lengths:
        print("No BPSEQ files found.")
        sys.exit(0)

    plt.figure()
    plt.hist(lengths, bins=30)
    plt.xlabel("Sequence length (nt)")
    plt.ylabel("Count")
    plt.title("BPSEQ Sequence Length Distribution")
    plt.tight_layout()
    plt.show()