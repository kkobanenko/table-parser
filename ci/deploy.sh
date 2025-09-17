#!/bin/bash
set -e

cd /home/$USER/table_parser   # каталог на сервере, где лежит docker-compose.yml

# обновляем образы и перезапускаем сервисы
docker compose pull
docker compose up -d

