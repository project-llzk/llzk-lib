// SPDX-License-Identifier: MIT
/**

Doxygen Awesome
https://github.com/jothepro/doxygen-awesome-css

Copyright (c) 2022 - 2025 jothepro

Modified by Veridise to work with hidden TOC levels.
*/

class InteractiveToc {
  static topOffset = 38;
  static hideMobileMenu = true;
  static headers = [];

  static init(defaultLevel) {
    window.addEventListener("load", () => {
      const meta = document.querySelector('meta[name="toc-level"]');
      const level = meta ? parseInt(meta.content, 10) : defaultLevel;

      let toc = document.querySelector(".contents > .toc");
      if (toc) {
        toc.classList.add("interactive");
        if (!InteractiveToc.hideMobileMenu) {
          toc.classList.add("open");
        }
        document
          .querySelector(".contents > .toc > h3")
          ?.addEventListener("click", () => {
            if (toc.classList.contains("open")) {
              toc.classList.remove("open");
            } else {
              toc.classList.add("open");
            }
          });

        // Helper function to recursively hide entries deeper than `level`
        function hideDeepEntries() {
          document.querySelectorAll(".contents > .toc li").forEach((li) => {
            const classes = li.className.split(/\s+/);
            const levelClass = classes.find((c) => /^level\d+$/.test(c));
            if (levelClass) {
              const liLevel = parseInt(levelClass.replace("level", ""), 10);
              if (liLevel > level) {
                li.style.display = "none"; // hide if deeper than max level
              }
            }
          });
        }
        // Apply hiding before processing headers
        const topUl = toc.querySelector("ul");
        hideDeepEntries();

        document.querySelectorAll(".contents > .toc li a").forEach((node) => {
          // Ignore hidden headers.
          if (node.closest("li").style.display === "none") return;

          let id = node.getAttribute("href").substring(1);
          let header = {
            node: node,
            headerNode: document.getElementById(id),
          };
          InteractiveToc.headers.push(header);

          document
            .getElementById("doc-content")
            ?.addEventListener(
              "scroll",
              this.throttle(InteractiveToc.update, 100),
            );
        });
        InteractiveToc.update();
      }
    });
  }

  static update() {
    let active = InteractiveToc.headers[0]?.node;
    InteractiveToc.headers.forEach((header) => {
      let position = header.headerNode?.getBoundingClientRect().top;
      if (position !== undefined) {
        header.node.classList.remove("active");
        header.node.classList.remove("aboveActive");
        if (position < InteractiveToc.topOffset) {
          active = header.node;
          active?.classList.add("aboveActive");
        }
      }
    });
    active?.classList.add("active");
    active?.classList.remove("aboveActive");
  }

  static throttle(func, delay) {
    let lastCall = 0;
    return function (...args) {
      const now = new Date().getTime();
      if (now - lastCall < delay) {
        return;
      }
      lastCall = now;
      return setTimeout(() => {
        func(...args);
      }, delay);
    };
  }
}
