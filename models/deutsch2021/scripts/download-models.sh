mkdir -p models

if [ "$QG" = "true" ]; then
  gdown https://drive.google.com/uc?id=1vVhRgLtsQDAOmxYhY5PMPnxxHUyCOdQU
  mv model.tar.gz models/question-generation.model.tar.gz
fi

if [ "$QA" = "true" ]; then
  mkdir -p models/question-answering
  cd models/question-answering
  gdown https://drive.google.com/uc?id=1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8
  unzip model.zip
  rm model.zip
fi
