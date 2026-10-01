# Code Changes - Keishana Morsley (1001807020)
— Removed `load_data(path)` helper
— What I changed: Deleted the `load_data` function and replaced its single call with `pd.read_csv(data_path)` directly in `main()`.
— Why: The helper only wrapped a single pandas call and was not reused elsewhere, adding unnecessary indirection.
— Benefit: Simplifies control flow and makes the code easier to follow.

— Replaced manual loop-based summary with pandas aggregation
— What I changed: Removed the `compute_summary` function that iterated over columns and computed statistics per column; instead I used `summary_df = df[cols].agg(['mean', 'median', 'min', 'max', 'std']).T`.
— Why: The original loop was longer than necessary and repeated pandas operations.
— Benefit: Uses a cleaner pandas aggregation approach, reduces lines of code, and keeps the same results.

— Removed `count_performance_categories(df)` helper
— What I changed: Deleted the helper and used `perf_counts = df['Performance_Category'].value_counts()` directly in `main()`.
— Why: It was a single-line wrapper that did not improve reuse or readability.
— Benefit: Reduces unnecessary indirection and keeps simple logic where it is used.

— Removed `save_summary` and `save_counts` helpers
— What I changed: Deleted the separate saving functions and used `to_csv()` directly where the output files are created.
— Why: These functions only forwarded data to pandas `to_csv()` and did not add any additional processing.
— Benefit: Removes repetitive code and makes the saving steps more direct.

— Simplified plotting helper and removed backend switching
— What I changed: Replaced the larger `plot_and_save_pie` function with a shorter `save_pie_chart(counts, path)` function and removed `plt.switch_backend('Agg')`.
— Why: The backend switch was unnecessary for saving the static chart in this environment, and the plotting logic could be shorter.
— Benefit: Keeps the plotting code organized while making it easier to read.

— Simplified file path handling
— What I changed: Used `Path(__file__).parent` instead of `Path(__file__).resolve().parent` and used clearer variable names such as `data_path`, `summary_path`, `counts_path`, and `pie_path`.
— Why: The simpler path approach is sufficient for files stored in the assignment folder.
— Benefit: Makes the path-handling code shorter and easier to understand.

— Improved variable names and reduced repetition
— What I changed: Used clearer variable names such as `cols`, `summary_df`, `perf_counts`, and `pie_path`, while removing unnecessary forwarding functions.
— Why: Clear names make the purpose of each variable easier to understand.
— Benefit: Improves readability and maintainability without changing the program’s behavior.

— Verified all required outputs and displays
— What I verified: The script still prints the first five rows, calculates mean, median, minimum, maximum, and standard deviation for `Study_Hours`, `Attendance_Pct`, and `Exam_Score`, counts `Performance_Category`, saves `summary_statistics.csv`, saves `performance_category_counts.csv`, and creates `performance_category_pie.png`.
— Why: I wanted to make sure the cleanup did not remove any required assignment functionality.
— Benefit: Confirms the simplified script still meets all Assignment 1 requirements.
