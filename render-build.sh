#!/usr/bin/env bash
# Install system dependencies
apt-get update && apt-get install -y build-essential libpq-dev
# Install Python dependencies
pip install -r requirements.txt
# Run database migrations
python manage.py migrate
# Collect static files
python manage.py collectstatic --noinput
