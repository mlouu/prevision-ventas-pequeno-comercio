# Contraste del árbol de supervivencia de src/supervivencia.py con
# rpart(method = "exp") de R.
#
# Uso, desde la raíz del proyecto:
#   .venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); import supervivencia as sv; sv.exportar_para_rpart(sv.construir_duraciones(), 'duraciones.csv')"
#   Rscript src/supervivencia_rpart.R duraciones.csv
#
# Mismos límites que ArbolSupervivencia: profundidad 3 y 20 artículos por
# hoja como mínimo. cp = 0 para que no pode por su cuenta, xval = 0 porque la
# validación se hace en Python con los mismos pliegues que el resto.
#
# Resultado del 2026-09-28: las mismas 7 preguntas, los mismos umbrales y las
# mismas 8 hojas con el mismo número de artículos. Los riesgos relativos
# difieren en centésimas por la contracción bayesiana que rpart aplica a la
# tasa de cada hoja y la versión de Python no.

library(rpart)
library(survival)

d <- read.csv(commandArgs(trailingOnly = TRUE)[1])
m <- rpart(Surv(tiempo, evento) ~ ., data = d, method = "exp",
           control = rpart.control(maxdepth = 3, minbucket = 20, cp = 0, xval = 0))
print(m)
