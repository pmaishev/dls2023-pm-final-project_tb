# Telegram Bot
Final project for the course [Deep Learning (Semester 1, Spring 2023): Advanced Track](https://stepik.org/course/135003/syllabus)

## Project Structure
```
bot
│   .gitignore
│   .gitlab-ci.yml
│   app.py
│   config.py
│   README.md
│   requirements.txt
│
├───.github
│   └───workflows
│           docker-image.yml
│
├───build
│       Dockerfile
│       DockerfileBase
│
├───cnn
│       vgg19.pth
│       msgnet_21_styles.pth
│
├───data_test
│   ├───0
│   │       content_bruges.jpg
│   │       result_50.png
│   │       style_starry_night.jpg
│   │
│   ├───1
│   │       content_IMG_7359.jpg
│   │       result_50.png
│   │       style_IMG_7580.jpg
│   │
│   └───2
│           dancing.jpg
│           picasso.jpg
│           result_50.png
│
├───model
│       style_transfer.py
│       style_transfer_msg.py
│
└───tests
        test.py

```
### Root Folder
The root directory contains the following files:
1. `app.py` - The main project file containing the bot.
2. `config.py` - Configuration and message templates for the bot.
3. `requirements.txt` - Required libraries.
4. `README.md` - Project description (this file).
5. `.gitignore` - List of files and directories to ignore in Git.
6. `.gitlab-ci.yml` - CI/CD configuration for GitLab.

### .github folder
Github Actions for CI/CD on GitHub.

### build Folder
1. `Dockerfile` - Dockerfile for the bot.
2. `DockerfileBase` - Base Dockerfile from which the bot is built. Created to avoid rebuilding everything from scratch with each commit.

### cnn Folder
1. `vgg19.pth` - Saved VGG19 model with weights to avoid reloading it every time the bot is updated.
2. `msgnet_21_styles.pth` - Saved weights for MSG Net.

### data_test Folder
Test data.

### model Folder
1. `style_transfer.py` - Class for style transfer and its configuration.
2. `style_transfer_msg.py` - Classes for fast style transfer and its configuration. https://arxiv.org/abs/1703.06953

### tests Folder
Tests.

### Bot Commands
1. `/start` - Displays a brief description of the bot.
2. `/help` - Similar to the previous command.
3. `/style_transfer` - Requests two images and transfers the style from one to the other using MSG Net.
4. `/style_transfer_slow` - Requests two images and transfers the style from one to the other.

### Authors and Acknowledgment
- Author: Petr Maishev
- Email: pmaishev@gmail.com
- StepikId: 82457743
- Telegram Id: @pmaishev

### Roadmap
TODO: Convert model to ONNX
TODO: Fit model to my own styles
