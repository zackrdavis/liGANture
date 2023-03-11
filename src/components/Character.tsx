import styled from "styled-components";
import * as ort from "onnxruntime-web";
import { useEffect, useRef } from "react";
import { drawOutputToCanvas } from "../utils";

const CharWrap = styled.div`
  position: relative;
  display: inline-block;
  width: 100px;
  height: 100px;
`;

const CharCanvas = styled.canvas`
  width: 100px;
  height: 100px;
  // rotate and mirror
  transform: rotate(90deg) scaleY(-1);
`;

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
