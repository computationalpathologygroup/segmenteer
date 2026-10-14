const SITE = {
  repositoryUrl: "https://github.com/computationalpathologygroup/segmenteer",
  version: "0.1.0",
};


function initRepositoryLinks() {
  document.querySelectorAll("[data-repo-link]").forEach((link) => {
    link.href = SITE.repositoryUrl;
    link.target = "_blank";
    link.rel = "noreferrer";
  });
  document.querySelectorAll("[data-version]").forEach((node) => {
    node.textContent = SITE.version;
  });
}

function initNavigation() {
  const header = document.querySelector("[data-header]");
  const button = document.querySelector("[data-menu-button]");
  if (!header || !button) return;

  button.addEventListener("click", () => {
    const isOpen = header.dataset.open === "true";
    header.dataset.open = String(!isOpen);
    button.setAttribute("aria-expanded", String(!isOpen));
  });

  header.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      header.dataset.open = "false";
      button.setAttribute("aria-expanded", "false");
    });
  });
}

function initBoundaryStudy() {
  const figure = document.querySelector("[data-boundary-study]");
  const svg = figure?.querySelector(".study-svg");
  if (!figure || !svg) return;

  const speckleGroup = figure.querySelector(".specimen-speckles");
  if (!speckleGroup) return;

  const fragment = document.createDocumentFragment();
  let seed = 17;
  const random = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };

  for (let i = 0; i < 90; i += 1) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", String(115 + random() * 520));
    circle.setAttribute("cy", String(100 + random() * 400));
    circle.setAttribute("r", String(1 + random() * 4));
    fragment.appendChild(circle);
  }
  speckleGroup.appendChild(fragment);
}





initRepositoryLinks();
initNavigation();
initBoundaryStudy();
