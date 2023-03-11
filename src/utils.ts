import * as ort from "onnxruntime-web";
import { useEffect, useRef } from "react";

function clamp(input: number, min: number, max: number): number {
  return input < min ? min : input > max ? max : input;
}

export function projectRange(
  current: number,
  in_min: number,
  in_max: number,
  out_min: number,
  out_max: number
): number {
  const mapped: number =
    ((current - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min;
  return clamp(mapped, out_min, out_max);
}

export const drawOutputToCanvas = (
  nnOutput: ort.InferenceSession.OnnxValueMapType,
  ctx: CanvasRenderingContext2D
) => {
  const imgData = ctx.createImageData(28, 28);

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

/**
 * Feed the generator 100 random numbers, get 28*28 greyscale values
 * @param arr
 * @returns
 */
export const imgFromArray = async (
  arr: number[],
  session: ort.InferenceSession
) => {
  // convert array to tensor
  const tensor = new ort.Tensor("float32", arr, [1, 100]);

  // run the generator
  var { img } = await session.run({ z: tensor });

  // map the raw outputs from -1 - 1 range to 0 - 255 range
  return Uint8Array.from(img.data as Float32Array, (val) =>
    projectRange(val, -1, 1, 0, 255)
  );
};

/**
 * Like useEffect, but doesn't fire on render
 * @param effect
 * @param deps
 */
export const useUpdateEffect: typeof useEffect = (effect, deps) => {
  const isFirstMount = useRef(true);

  useEffect(() => {
    if (!isFirstMount.current) effect();
    else isFirstMount.current = false;
  }, deps);
};

export const isAlphaNum = (ch: string) =>
  ch.match(/^[a-z0-9]+$/i) !== null && ch.length == 1;
