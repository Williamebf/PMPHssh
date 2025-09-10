import random

def generate_i32_array_file(filename: str, count: int, batch_size: int = 100_000) -> None:
    """
    Creates a file containing a single-line Python-style array
    with `count` random integers between -50 and 50, each suffixed with 'i32'.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('[')
        first_batch = True
        full_batches = count // batch_size
        remainder = count % batch_size

        for _ in range(full_batches):
            # Generate and format a batch of random i32 values
            nums = [f"{random.randint(-128, 128)}i32" for _ in range(batch_size)]
            chunk = ", ".join(nums)
            if first_batch:
                f.write(chunk)
                first_batch = False
            else:
                f.write(", " + chunk)

        if remainder:
            nums = [f"{random.randint(-128, 128)}i32" for _ in range(remainder)]
            chunk = ", ".join(nums)
            if first_batch:
                f.write(chunk)
            else:
                f.write(", " + chunk)

        f.write(']')

if __name__ == "__main__":
    TOTAL_COUNT = 9000000
    OUTPUT_FILE = "random3_i32_array.txt"
    generate_i32_array_file(OUTPUT_FILE, TOTAL_COUNT)
    print(f"Wrote {TOTAL_COUNT} elements to '{OUTPUT_FILE}'")
