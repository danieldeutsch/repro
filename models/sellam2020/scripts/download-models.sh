if [ "$TINY_128" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip
  unzip bleurt-tiny-128.zip
  rm bleurt-tiny-128.zip
fi

if [ "$TINY_512" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip
  unzip bleurt-tiny-512.zip
  rm bleurt-tiny-512.zip
fi

if [ "$BASE_128" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
  unzip bleurt-base-128.zip
  rm bleurt-base-128.zip
fi

if [ "$BASE_512" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip
  unzip bleurt-base-512.zip
  rm bleurt-base-512.zip
fi

if [ "$LARGE_128" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip
  unzip bleurt-large-128.zip
  rm bleurt-large-128.zip
fi

if [ "$LARGE_512" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip
  unzip bleurt-large-512.zip
  rm bleurt-large-512.zip
fi
