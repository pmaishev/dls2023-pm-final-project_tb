# Telegram bot
Итоговый проект курса [Deep Learning (семестр 1, весна 2023): продвинутый поток](https://stepik.org/course/135003/syllabus)

## Annotation.
[[_TOC_]]

## Структура проекта
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
│
└───tests
        test.py

```
### root folder
В корневой директории находятся следующие файлы:
1. `app.py` - основной файл проекта с ботом
1. `config.py` - конфигурация и шаблоны сообщений бота
1. `requirements.txt` - необходимые библиотеки
1. `README.md` - описание проекта (этот файл)
1. `.gitignore` - список того, что не сохраняем в гит
1. `.gitlab-ci.yml` - настройка CI/CD для GitLab

### .github folder
Github actions для CI/CD на github

### build folder
1. `Dockerfile` - dockerfile для бота
1. `DockerfileBase` - базовый dockerfile, от которого строится бот. Сделан для того, чтобы не пересобирать все с нуля при каждом коммите.

### cnn folder
1. `vgg19.pth` - сохраненный VGG19 с весами, чтобы не загружать каждый раз при обновдении бота.

### data_test folder
Тестовые данные.

### model folder
1. `style_transfer.py` класс для переноса стиля и его конфигурация.

### tests folder
Тесты.

## Команды бота
1. `/start` - выводит краткое описание бота.
1. `/help` - аналогично предыдущей команде.
1. `style_transfer` - запращивает два изображения и переносит стиль с одного на другое.

## Roadmap
TODO: If you have ideas for releases in the future, it is a good idea to list them in the README.

## Authors and acknowledgment
Автор: Petr Maishev
Email: pmaishev@gmail.com
StepikId: 82457743
Telegram Id: @pmaishev

## Roadmap
TODO: If you have ideas for releases in the future, it is a good idea to list them in the README.

## License
MIT License

Copyright (c) 2023 Petr Maishev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
