import { pathToRoot, joinSegments, isAbsoluteURL } from "../util/path"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

interface Options {
  links: Record<string, string>
}

export default ((opts?: Options) => {
  const NavLinks: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const baseDir = pathToRoot(fileData.slug!)
    const links = opts?.links ?? {}

    return (
      <nav class={classNames(displayClass, "nav-links")}>
        <ul>
          {Object.entries(links).map(([label, link]) => {
            const href = isAbsoluteURL(link) ? link : joinSegments(baseDir, link)
            return (
              <li>
                <a href={href}>{label}</a>
              </li>
            )
          })}
        </ul>
      </nav>
    )
  }

  NavLinks.css = `
.nav-links ul {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  list-style: none;
  padding: 0;
  margin: 0;
}

.nav-links a {
  color: var(--darkgray);
}

.nav-links a:hover {
  color: var(--secondary);
}
`

  return NavLinks
}) satisfies QuartzComponentConstructor<Options | undefined>
