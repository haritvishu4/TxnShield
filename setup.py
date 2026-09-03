from setuptools import setup, find_packages

setup(
    name="fraud_detection_system",
    version="1.0.0",
    author="Bhavishya Sharma",
    description="Real-Time Fraud Detection and Risk Intelligence System",
    packages=find_packages(include=["src", "src.*", "api", "api.*", "dashboard", "dashboard.*"]),
    python_requires=">=3.10",
)
