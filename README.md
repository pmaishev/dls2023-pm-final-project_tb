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
1. `msgnet_21_styles.pth` - сохраненные веса MSG Net.

### data_test folder
Тестовые данные.

### model folder
1. `style_transfer.py` класс для переноса стиля и его конфигурация.
1. `style_transfer_msg.py` классы для быстрого переноса стиля и его конфигурация. https://arxiv.org/abs/1703.06953

### tests folder
Тесты.

## Команды бота
1. `/start` - выводит краткое описание бота.
1. `/help` - аналогично предыдущей команде.
1. `style_transfer` - запращивает два изображения и переносит стиль с одного на другое при помощи MSG Net.
1. `style_transfer_slow` - запращивает два изображения и переносит стиль с одного на другое.

## Roadmap
TODO: If you have ideas for releases in the future, it is a good idea to list them in the README.

## Authors and acknowledgment
Автор: Petr Maishev
Email: pmaishev@gmail.com
StepikId: 82457743
Telegram Id: @pmaishev

## Roadmap
TODO: If you have ideas for releases in the future, it is a good idea to list them in the README.
