import style from "./styles/listPage.scss"
import { PageList, byDateAndAlphabetical } from "./PageList"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { concatenateResources } from "../util/resources"

const excludedSlugs = new Set(["index", "about", "archive", "tags"])

export default (() => {
  const AllArticles: QuartzComponent = (props: QuartzComponentProps) => {
    const pages = props.allFiles.filter((file) => {
      const slug = file.slug ?? ""
      return !excludedSlugs.has(slug) && !slug.endsWith("/index") && !slug.startsWith("tags/")
    })

    return (
      <div class="page-listing">
        <p>共 {pages.length} 篇文章</p>
        <PageList {...props} allFiles={pages} sort={byDateAndAlphabetical(props.cfg)} />
      </div>
    )
  }

  AllArticles.css = concatenateResources(style, PageList.css)
  return AllArticles
}) satisfies QuartzComponentConstructor
