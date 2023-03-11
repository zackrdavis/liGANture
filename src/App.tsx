import { KeyboardEventHandler, useEffect, useRef, useState } from "react";
import { Character } from "./components/Character";
import * as ort from "onnxruntime-web";
import { isAlphaNum } from "./utils";
import { addresses } from "./components/addresses";
import styled from "styled-components";
import { useOnnxSession } from "./useOnnxSession";
import { Tensor } from "onnxruntime-web";

//TODO: Type THESE
import { useInterval } from "./components/useInterval";
import * as lerp_array from "lerp-array";

const AppWrap = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  padding: 20px;
`;

type LetterForm = {
  char: string[];
  currAddr?: number[];
  image: ort.InferenceSession.OnnxValueMapType;
};

function App() {
  const appRef = useRef<HTMLDivElement>(null);
  const [chars, setChars] = useState<LetterForm[]>([]);
  const [heldKeys, setHeldKeys] = useState<Set<string>>(new Set([]));
  const destAddress = useRef<number[]>();
  const didArrive = useRef(false);

  // focus the typing area immediately
  useEffect(() => appRef.current?.focus());

  // create ONNX session
  const { requestInference } = useOnnxSession("./emnist_vgan.onnx");

  // setup animation interval for when keys are held
  useInterval(handleHeldKeys, heldKeys.size ? 100 : null);

  const handleKeyDown: KeyboardEventHandler = async (e) => {
    // reset didArrive whenever a new key is pressed
    if (![...heldKeys].includes(e.key)) {
      didArrive.current = false;
    }

    // update heldKeys
    if (isAlphaNum(e.key)) {
      const newHeldKeys = new Set([...heldKeys]);
      newHeldKeys.add(e.key);
      setHeldKeys(newHeldKeys);
    }

    // handle non-letter keys
    if (e.key == "Backspace") {
      doBackspace(chars, setChars);
    } else if (e.key == " ") {
      doSpace(chars, setChars);
    } else if (isAlphaNum(e.key) && !heldKeys.size) {
      doLetter(e.key, chars, setChars, requestInference);
    }
    // TODO: LINEBREAK
    // TODO: HANDLE SHIFT+KEY
  };

  const handleKeyUp: KeyboardEventHandler = (e) => {
    const newHeldKeys = new Set([...heldKeys]);
    newHeldKeys.delete(e.key);
    setHeldKeys(newHeldKeys);
    didArrive.current = false;
  };

  async function handleHeldKeys() {
    const heldKeyArr = [...heldKeys];
    const currAddr = chars[chars.length - 1].currAddr;

    // check if we have arrived at destAddress
    if (currAddr?.toString() == destAddress?.current?.toString()) {
      console.log("arrived");
      didArrive.current = true;
    }

    if (heldKeyArr.length == 1) {
      // destAddress corresponds to a single letter
      destAddress.current = addresses[heldKeyArr[0]];
    } else {
      // gather addresses to average
      const heldAddrs = heldKeyArr.map((key) => addresses[key]);

      // generate mean address
      let result = [];
      for (let i = 0; i < heldAddrs[0].length; i++) {
        let num = 0;
        for (let j = 0; j < heldAddrs.length; j++) {
          num += heldAddrs[j][i];
        }
        result.push(num / heldAddrs.length);
      }

      // set mean as destination
      destAddress.current = result;
    }

    if (currAddr) {
      if (!didArrive.current) {
        // lerp toward the destination address
        const incr = 0.1;
        const newAddr = [];
        for (const [i, curr] of currAddr.entries()) {
          const dest = destAddress.current[i];
          const diff = dest - curr;

          if (Math.abs(diff) <= incr) {
            // if increment would go past dest, go to dest
            newAddr.push(dest);
          } else {
            // otherwise, increment toward dest
            newAddr.push(curr + Math.sign(diff) * incr);
          }
        }

        const newLastChar: LetterForm = {
          char: heldKeyArr,
          currAddr: newAddr,
          image: await requestInference(newAddr),
        };

        setChars([...chars.slice(0, -1), newLastChar]);
      } else {
        // drunkenly explore around current address
        console.log("explore");
        const newAddr = [];
        for (const num of currAddr) {
          const incr = Math.random() < 0.5 ? -0.05 : 0.05;
          newAddr.push(num + incr);
        }

        const newLastChar: LetterForm = {
          char: heldKeyArr,
          currAddr: newAddr,
          image: await requestInference(newAddr),
        };

        setChars([...chars.slice(0, -1), newLastChar]);
      }
    }
  }

  const doExploreNear = async (
    key: string,
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void,
    requestInference: (
      address: number[]
    ) => Promise<ort.InferenceSession.OnnxValueMapType>
  ) => {
    console.log("holding same");
  };

  const doBackspace = (
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void
  ) => {
    setChars(chars.slice(0, -1));
  };

  const doSpace = (
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void
  ) => {
    setChars([
      ...chars,
      {
        char: [" "],
        image: { img: new Tensor("float32", Array(784).fill(-1), [1, 784]) },
      },
    ]);
  };

  const doLetter = async (
    key: string,
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void,
    requestInference: (
      address: number[]
    ) => Promise<ort.InferenceSession.OnnxValueMapType>
  ) => {
    setChars([
      ...chars,
      {
        char: [key],
        currAddr: addresses[key],
        image: await requestInference(addresses[key]),
      },
    ]);
  };

  const doHybridLetter = async (
    key: string,
    chars: LetterForm[],
    setChars: (a: LetterForm[]) => void,
    requestInference: (
      address: number[]
    ) => Promise<ort.InferenceSession.OnnxValueMapType>
  ) => {
    console.log("holding multiple");

    const addressesToCombine = [
      [...addresses[chars[chars.length - 1].char[0]]],
      [...addresses[key]],
    ];

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

  return (
    <div className="App">
      <AppWrap
        ref={appRef}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
      >
        {chars.map((c, i) => (
          <Character key={i} chars={c.char} nnOutput={c.image} />
        ))}
      </AppWrap>
    </div>
  );
}

export default App;
