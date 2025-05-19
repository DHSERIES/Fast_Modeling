# Makefile for Python project

# Define the virtual environment directory and requirements file
VENV_DIR=.venv
REQUIREMENTS=requirements.txt

# Python interpreter
PYTHON=python3


# Create the virtual environment
$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies from requirements.txt
install: $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)

# Run the application (e.g., Streamlit)
run:
	$(VENV_DIR)/bin/streamlit run main.py

create_dirs:
	@mkdir -p data report
	@echo "Directories and beginner main file created."


# Default target to create the virtual environment and install dependencies
all: install create_dirs run