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

if [ "$BLEURT_20" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
  unzip BLEURT-20.zip
  rm BLEURT-20.zip
fi

if [ "$BLEURT_20_D12" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
  unzip BLEURT-20-D12.zip
  rm BLEURT-20-D12.zip
fi

if [ "$BLEURT_20_D6" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip
  unzip BLEURT-20-D6.zip
  rm BLEURT-20-D6.zip
fi

if [ "$BLEURT_20_D3" = "true" ]; then
  wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip
  unzip BLEURT-20-D3.zip
  rm BLEURT-20-D3.zip
fi
