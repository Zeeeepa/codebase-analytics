import React, { useState, useRef, useEffect } from 'react';

interface TooltipProps {
  children: React.ReactNode;
  content: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
}

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  position = 'top',
  delay = 300,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    timerRef.current = setTimeout(() => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        
        let x = 0;
        let y = 0;
        
        switch (position) {
          case 'top':
            x = rect.left + rect.width / 2;
            y = rect.top;
            break;
          case 'bottom':
            x = rect.left + rect.width / 2;
            y = rect.bottom;
            break;
          case 'left':
            x = rect.left;
            y = rect.top + rect.height / 2;
            break;
          case 'right':
            x = rect.right;
            y = rect.top + rect.height / 2;
            break;
        }
        
        setCoords({ x, y });
        setIsVisible(true);
      }
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const getTooltipStyle = () => {
    if (!tooltipRef.current) return {};
    
    const tooltipRect = tooltipRef.current.getBoundingClientRect();
    
    let style: React.CSSProperties = {
      position: 'fixed',
      zIndex: 1000,
    };
    
    switch (position) {
      case 'top':
        style = {
          ...style,
          left: coords.x - tooltipRect.width / 2,
          bottom: window.innerHeight - coords.y + 8,
        };
        break;
      case 'bottom':
        style = {
          ...style,
          left: coords.x - tooltipRect.width / 2,
          top: coords.y + 8,
        };
        break;
      case 'left':
        style = {
          ...style,
          right: window.innerWidth - coords.x + 8,
          top: coords.y - tooltipRect.height / 2,
        };
        break;
      case 'right':
        style = {
          ...style,
          left: coords.x + 8,
          top: coords.y - tooltipRect.height / 2,
        };
        break;
    }
    
    return style;
  };

  return (
    <div
      ref={containerRef}
      className="inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {isVisible && (
        <div
          ref={tooltipRef}
          className="bg-gray-800 text-white text-xs rounded py-1 px-2 pointer-events-none"
          style={getTooltipStyle()}
        >
          {content}
          <div
            className={`absolute w-2 h-2 bg-gray-800 transform rotate-45 ${
              position === 'top' ? 'top-full -mt-1 left-1/2 -ml-1' :
              position === 'bottom' ? 'bottom-full -mb-1 left-1/2 -ml-1' :
              position === 'left' ? 'left-full -ml-1 top-1/2 -mt-1' :
              'right-full -mr-1 top-1/2 -mt-1'
            }`}
          />
        </div>
      )}
    </div>
  );
};

