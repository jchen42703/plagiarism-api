# Plagiarism Detection Spec

## Check if an image is plagiarized

`POST /verify`

```
{
    image: "...",
    thresholdConfidence?: number between 0-1
}
```

- `image` should be a base64 encoding of an image
- `thresholdConfidence` is the threshold that we would consider an image plagiarized.
  - The higher the threshold, the more strict the detection must be

```
{
    "likelihood": number between 0-1,
    "matchedImage": string
}
```

- `likelihood` is the confidence that the model thinks the image is plagiarized
- `matchedImage` should be a base64 encoding of an image

## Check
