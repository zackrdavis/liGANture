import { useRef, useEffect } from "react";

// https://www.joshwcomeau.com/snippets/react-hooks/use-interval/
export function useInterval(callback: () => void, delay: number | null) {
  const intervalRef = useRef(setInterval(() => {}, Infinity));
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    const tick = () => savedCallback.current();

    if (typeof delay === "number") {
      intervalRef.current = setInterval(tick, delay);
      return () => clearInterval(intervalRef.current);
    }
  }, [delay]);

  return intervalRef;
}
