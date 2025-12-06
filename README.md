# iris_cryptotrader
trading algoritmico
cd C:\Iris
gh repo create cristian-armijo/iris_cryptotrader --public --source=. --remote=origin --push

cd C:\Iris
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
# o por SSH
git remote set-url origin git@github.com:TU_USUARIO/TU_REPOSITORIO.git

git branch -M main
git push -u origin main

git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@example.com"

git checkout -b feature/mi-cambio

git checkout main
git pull origin main