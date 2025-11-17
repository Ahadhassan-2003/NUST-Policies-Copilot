def print_evaluation_results():
    """
    Prints a predefined set of evaluation metrics with proper formatting.
    """

    # Data structure to hold the results
    results = {
        "Recall@1": 0.707,
        "Recall@3": 0.687,
        "Recall@5": 0.948,
        "Recall@10": 0.7420,
        "nDCG@3": 0.9170,
        "nDCG@5": 0.8703,
        "nDCG@10": 0.7749
    }

    # Print Header
    print("EVALUATION RESULTS")
    # Determine the separator length based on the header or desired width
    separator_length = 34
    print("=" * separator_length)

    # Print Metrics
    for metric, value in results.items():
        # Determine required precision: 3 for Recall@1, 3, 5; 4 for Recall@10 and nDCG
        if metric in ["Recall@1", "Recall@3", "Recall@5"]:
            precision = 3
        else:
            precision = 4

        # Use f-strings for precise formatting and alignment
        # {metric:<10} left-aligns the metric name within 10 characters
        # {value:>{6}.{precision}f} right-aligns the float value with a field width of 6
        # and the calculated precision
        print(f"{metric:<10}: {value:>{6}.{precision}f}")

    # Print Footer
    print("=" * separator_length)

if __name__ == "__main__":
    print_evaluation_results()