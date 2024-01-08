import numpy as np


def execution_time_report(times_list: list, title: str = "Execution time", unit: str = "ms", tails: bool = True):
    if not tails:
        times_list = times_list[1:-1]

    match unit:
        case "ms" | "milliseconds":
            multiplier = 1_000
        case "us" | "microseconds":
            multiplier = 1_000_000
        case "ns" | "nanoseconds":
            multiplier = 1_000_000_000
        case _:
            unit = "s"
            multiplier = 1

    print(f"--- {title} ---\n"
          f" - Mean : {round(np.mean(times_list) * multiplier, 3)} ±{round(np.std(times_list)*1000,3)} {unit}\n"
          f" - Max: {round(np.max(times_list) * multiplier, 3)} {unit}\n"
          f" - Min: {round(np.min(times_list) * multiplier, 3)} {unit}\n")


if __name__ == "__main__":
    execution_time_report(list(range(11)), unit="us")
    execution_time_report(list(range(11)), tails=False)
