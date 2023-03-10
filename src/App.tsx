import {
  KeyboardEventHandler,
  MouseEventHandler,
  useEffect,
  useRef,
  useState,
} from "react";
import { Character } from "./components/Character";
import * as ort from "onnxruntime-web";
import { imgFromArray, isAlphaNum } from "./utils";
import { useUpdateEffect } from "./utils";
import { addresses } from "./components/addresses";
import styled from "styled-components";
import * as lerp_array from "lerp-array";
import { useOnnxSession } from "./useOnnxSession";

const AppWrap = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

const addHybridLetter = async (
  key: string,
  chars: LetterForm[],
  setChars: (a: LetterForm[]) => void,
  requestInference: (
    address: number[]
  ) => Promise<ort.InferenceSession.OnnxValueMapType>
) => {
  const addressesToCombine = [
    [...addresses[chars[chars.length - 1].char[0]]],
    [...addresses[key]],
  ];

  console.log(addressesToCombine[0]);
  console.log(addressesToCombine[1]);

  for (const step of [
    0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
  ]) {
    const hybridAddress = lerp_array(...addressesToCombine, step);

    const newLastChar: LetterForm = {
      char: [...chars[chars.length - 1].char, key],
      image: await requestInference(hybridAddress),
    };
    setChars([...chars.slice(0, -1), newLastChar]);
  }
};

type LetterForm = {
  char: string[];
  image: ort.InferenceSession.OnnxValueMapType;
};

function App() {
  const keyHeld = useRef(false);
  const appRef = useRef<HTMLDivElement>(null);
  const [chars, setChars] = useState<LetterForm[]>([]);

  const { requestInference } = useOnnxSession("./emnist_vgan.onnx");

  const handleKey: KeyboardEventHandler = async (e) => {
    if (e.key == "Backspace") {
      setChars(chars.slice(0, -1));
    } else if (!keyHeld.current) {
      // handle adding new character, disabled if still holding previous key
      keyHeld.current = true;

      if (isAlphaNum(e.key) || e.key == " ") {
        setChars([
          ...chars,
          {
            char: [e.key],
            image: await requestInference(addresses[e.key]),
          },
        ]);
      }
    } else if (
      // a letter key is already held
      isAlphaNum(e.key) && // new key is a alphanumeric
      isAlphaNum(chars[chars.length - 1].char[0]) && // previous chars alphanumeric
      !chars[chars.length - 1].char.includes(e.key) // previous chars doesn't contain the new letter
    ) {
      await addHybridLetter(e.key, chars, setChars, requestInference);
    }
  };

  useEffect(() => {
    appRef.current?.focus();
  });

  useEffect(() => {
    console.log(chars);
  }, [chars]);

  return (
    <div className="App">
      <AppWrap
        ref={appRef}
        tabIndex={0}
        onKeyDown={handleKey}
        onKeyUp={() => (keyHeld.current = false)}
      >
        {chars.map((c, i) => (
          <Character key={i} chars={c.char} nnOutput={c.image} />
        ))}
      </AppWrap>
    </div>
  );
}

export default App;
