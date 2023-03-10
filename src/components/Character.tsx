import styled from "styled-components";
import * as ort from "onnxruntime-web";
import { useEffect, useRef } from "react";
import { projectRange } from "../utils";

const CharWrap = styled.div`
  position: relative;
  display: inline-block;
  width: 100px;
  height: 100px;
`;

const CharCanvas = styled.canvas`
  width: 100px;
  height: 100px;
  transform: rotate(90deg) scaleY(-1);
`;

const drawOutputToCanvas = (
  nnOutput: ort.InferenceSession.OnnxValueMapType,
  ctx: CanvasRenderingContext2D
) => {
  const imgData = ctx.createImageData(28, 28);
  console.log(nnOutput);

  // map the raw outputs from -1 - 1 range to 0 - 255 range
  const asUInt8 = Uint8Array.from(nnOutput.img.data as Float32Array, (val) =>
    projectRange(val, -1, 1, 0, 255)
  );

  // fill RGB with original Luminance value (inverted)
  // Alpha channel is always 255
  for (let i = 0; i < imgData?.data.length; i += 4) {
    imgData.data[i + 0] = 255 - asUInt8[Math.floor(i / 4)];
    imgData.data[i + 1] = 255 - asUInt8[Math.floor(i / 4)];
    imgData.data[i + 2] = 255 - asUInt8[Math.floor(i / 4)];
    imgData.data[i + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
};

type CharacterProps = {
  nnOutput: ort.InferenceSession.OnnxValueMapType;
  chars: string[];
};

export const Character = ({ nnOutput, chars }: CharacterProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");

    if (nnOutput && ctx) {
      drawOutputToCanvas(nnOutput, ctx);
    }
  }, [nnOutput]);

  return (
    <CharWrap>
      <CharCanvas ref={canvasRef} width="28" height="28" />
      {/* <span style={{ position: "absolute", top: 0, left: 0 }}>{chars}</span> */}
    </CharWrap>
  );
};
