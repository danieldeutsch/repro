if [ "$TRANSFORMERABS_CNNDM" = "true" ]; then
  gdown https://drive.google.com/uc?id=1yLCqT__ilQ3mf5YUUCw9-UToesX5Roxy
  mv cnndm_baseline_best.pt transformerabs_cnndm.pt
fi

if [ "$BERTSUMEXT_CNNDM" = "true" ]; then
  gdown https://drive.google.com/uc?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ
  unzip bertext_cnndm_transformer.zip
  rm bertext_cnndm_transformer.zip
  mv bertext_cnndm_transformer.pt bertsumext_cnndm.pt
fi

if [ "$BERTSUMEXTABS_CNNDM" = "true" ]; then
  gdown https://drive.google.com/uc?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr
  unzip bertsumextabs_cnndm_final_model.zip
  rm bertsumextabs_cnndm_final_model.zip
  mv model_step_148000.pt bertsumextabs_cnndm.pt
fi

if [ "$BERTSUMEXTABS_XSUM" = "true" ]; then
  gdown https://drive.google.com/uc?id=1H50fClyTkNprWJNh10HWdGEdDdQIkzsI
  unzip bertsumextabs_xsum_final_model.zip
  rm bertsumextabs_xsum_final_model.zip
  mv model_step_30000.pt bertsumextabs_xsum.pt
fi