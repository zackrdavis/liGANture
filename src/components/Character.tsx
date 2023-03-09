import styled from "styled-components";
import * as ort from "onnxruntime-web";
import { imgFromArray, useUpdateEffect } from "../utils";
import { useEffect, useRef, useState } from "react";

const CharCanvas = styled.canvas`
  width: 100px;
  height: 100px;
`;

const drawFromArray = async (
  nnOutput: Uint8Array,
  ctx: CanvasRenderingContext2D
) => {
  const imgData = ctx.createImageData(28, 28);

  // fill RGB with original Luminance value (inverted)
  // Alpha channel is always 255
  for (let i = 0; i < imgData?.data.length; i += 4) {
    imgData.data[i + 0] = 255 - nnOutput[Math.floor(i / 4)];
    imgData.data[i + 1] = 255 - nnOutput[Math.floor(i / 4)];
    imgData.data[i + 2] = 255 - nnOutput[Math.floor(i / 4)];
    imgData.data[i + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
};

type CharacterProps = {
  sess: ort.InferenceSession;
  arr: number[];
};

export const Character = ({ sess, arr }: CharacterProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nnOutput, setNnOutput] = useState<Uint8Array>();
  const isModelRunning = useRef(false);

  useEffect(() => {
    if (!isModelRunning.current) {
      isModelRunning.current = true;
      imgFromArray(arr, sess).then((output) => {
        setNnOutput(output);
        isModelRunning.current = false;
      });
    }
  }, [arr]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");

    if (nnOutput && ctx) {
      drawFromArray(nnOutput, ctx);
    }
  }, [nnOutput]);

  return <CharCanvas ref={canvasRef} width="28" height="28" />;
};
