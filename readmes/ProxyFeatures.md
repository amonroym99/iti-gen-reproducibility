# Proxy features
This experiment shows how ITI-GEN uses gender as a proxy attribute when learning other attributes, such as "Bald", "Mustache" or "Beard". 

## Train ITI-GEN inclusive tokens
You have to run `jobfiles/proxy_features/iti_gen/generation/${attribute}.sh`. To run all of them
```shell
find jobfiles/proxy_features/iti_gen/generation/ -name "*.sh" | xargs -n1 bash
```

## Generation
You have to run `jobfiles/proxy_features/${method}/generation/${attribute}.sh`. To run all of them
```shell
find jobfiles/proxy_features/*/generation -name *.sh | xargs -n1 bash
```
