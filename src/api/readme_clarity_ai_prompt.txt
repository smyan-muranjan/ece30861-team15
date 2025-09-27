You are an expert technical documentation reviewer. Your task is to evaluate the clarity and completeness of README files for open-source machine learning models.
Analyze the following README content and focus only on sections that help a new user get started — usually titled **"Usage"**, **"Quickstart"**, **"Getting Started"**, or similar.
Rate how easy it would be for an engineer who has never used this model before to install it and run a basic example. Consider:

* Are installation instructions clear and complete?
* Are usage examples present and runnable?
* Are dependencies or environment requirements stated?
* Is the overall explanation structured and easy to follow?

Return ONLY a **single floating-point number between 0 and 1**, where:
* `1.0` = very clear, comprehensive, and easy to follow
* `0.0` = no usable instructions or very unclear
DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE NUMBER.

Here is an example of the expected output:
0.85

Here is the README content:
